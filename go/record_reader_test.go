//
// Copyright (c) 2025 ADBC Drivers Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package snowflake

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetArrowTimestampFromTimePrecision(t *testing.T) {
	values := []time.Time{
		time.Date(1, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2025, 6, 2, 10, 37, 56, 123456789, time.UTC),
		time.Date(9999, 12, 31, 23, 59, 59, 999999999, time.UTC),
	}
	for _, value := range values {
		t.Run(value.Format(time.RFC3339Nano), func(t *testing.T) {
			got, err := getArrowTimestampFromTime(value, arrow.Microsecond, arrow.Nanosecond, Microseconds)
			require.NoError(t, err)
			assert.Equal(t, arrow.Timestamp(value.UnixMicro()), got)

			got, err = getArrowTimestampFromTime(value, arrow.Nanosecond, arrow.Nanosecond, Nanoseconds)
			require.NoError(t, err)
			assert.Equal(t, arrow.Timestamp(value.UnixNano()), got)

			got, err = getArrowTimestampFromTime(value, arrow.Nanosecond, arrow.Nanosecond, NanosecondsNoOverflow)
			if value.Year() == 2025 {
				require.NoError(t, err)
				assert.Equal(t, arrow.Timestamp(value.UnixNano()), got)
			} else {
				require.ErrorContains(t, err, "overflows when converted to nanoseconds")
			}

			// A nanosecond precision limit must not promote lower-precision values.
			got, err = getArrowTimestampFromTime(value, arrow.Millisecond, arrow.Millisecond, Nanoseconds)
			require.NoError(t, err)
			assert.Equal(t, arrow.Timestamp(value.UnixMilli()), got)
		})
	}
}

// mockBatch implements batchStreamer for testing.
type mockBatch struct {
	streams []func(context.Context) (io.ReadCloser, error)
	call    int
	numRows int64
}

func (m *mockBatch) GetStream(ctx context.Context) (io.ReadCloser, error) {
	if m.call >= len(m.streams) {
		return nil, fmt.Errorf("no more streams configured")
	}
	fn := m.streams[m.call]
	m.call++
	return fn(ctx)
}

func (m *mockBatch) NumRows() int64 {
	return m.numRows
}

// buildIPCBytes writes Arrow IPC record batches to a byte buffer.
func buildIPCBytes(alloc memory.Allocator, schema *arrow.Schema, records []arrow.RecordBatch) []byte {
	var buf bytes.Buffer
	w := ipc.NewWriter(&buf, ipc.WithSchema(schema), ipc.WithAllocator(alloc))
	for _, rec := range records {
		_ = w.Write(rec)
	}
	_ = w.Close()
	return buf.Bytes()
}

func truncateIPCStream(data []byte) []byte {
	if len(data) <= 8 {
		return append([]byte(nil), data...)
	}
	return append([]byte(nil), data[:len(data)-8]...)
}

func testSchema() *arrow.Schema {
	return arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
	}, nil)
}

func buildTestRecord(alloc memory.Allocator, schema *arrow.Schema, values []int64) arrow.RecordBatch {
	bldr := array.NewRecordBuilder(alloc, schema)
	defer bldr.Release()
	for _, v := range values {
		bldr.Field(0).(*array.Int64Builder).Append(v)
	}
	return bldr.NewRecordBatch()
}

func identityTransform(_ context.Context, r arrow.RecordBatch) (arrow.RecordBatch, error) {
	r.Retain()
	return r, nil
}

func failingTransform(msg string) recordTransformer {
	return func(_ context.Context, r arrow.RecordBatch) (arrow.RecordBatch, error) {
		return nil, fmt.Errorf("%s", msg)
	}
}

func streamFromBytes(data []byte) func(context.Context) (io.ReadCloser, error) {
	return func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	}
}

func streamError(err error) func(context.Context) (io.ReadCloser, error) {
	return func(context.Context) (io.ReadCloser, error) {
		return nil, err
	}
}

type contextEOFStream struct {
	ctx context.Context

	prefix bytes.Reader

	blockOnce sync.Once
	blocked   chan struct{}
}

func newContextEOFStream(ctx context.Context, prefix []byte) *contextEOFStream {
	return &contextEOFStream{
		ctx:     ctx,
		prefix:  *bytes.NewReader(prefix),
		blocked: make(chan struct{}),
	}
}

func (s *contextEOFStream) Read(p []byte) (int, error) {
	if s.prefix.Len() > 0 {
		return s.prefix.Read(p)
	}
	s.blockOnce.Do(func() {
		close(s.blocked)
	})
	<-s.ctx.Done()
	return 0, io.EOF
}

func (s *contextEOFStream) Close() error {
	return nil
}

// --- tryReadBatch tests ---

func TestTryReadBatch_Success(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1, 2, 3})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 3, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()

	assert.EqualValues(t, 3, recs[0].NumRows())
	col := recs[0].Column(0).(*array.Int64)
	assert.EqualValues(t, 1, col.Value(0))
	assert.EqualValues(t, 2, col.Value(1))
	assert.EqualValues(t, 3, col.Value(2))
}

func TestTryReadBatch_MultipleRecords(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec1 := buildTestRecord(alloc, schema, []int64{10, 20})
	defer rec1.Release()
	rec2 := buildTestRecord(alloc, schema, []int64{30, 40})
	defer rec2.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec1, rec2})
	batch := &mockBatch{numRows: 4, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.NoError(t, err)
	require.Len(t, recs, 2)
	defer func() {
		for _, r := range recs {
			r.Release()
		}
	}()

	assert.EqualValues(t, 2, recs[0].NumRows())
	assert.EqualValues(t, 2, recs[1].NumRows())
}

func TestTryReadBatch_EmptyStream(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	data := buildIPCBytes(alloc, schema, nil) // no records, just schema
	batch := &mockBatch{numRows: 0, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.NoError(t, err)
	assert.Empty(t, recs)
}

func TestTryReadBatch_GetStreamError(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	batch := &mockBatch{streams: []func(context.Context) (io.ReadCloser, error){
		streamError(fmt.Errorf("network down")),
	}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "network down")
	assert.Nil(t, recs)
}

func TestTryReadBatch_TransformError(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 1, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, failingTransform("bad transform"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bad transform")
	// partial recs may be returned; caller is responsible for releasing them
	for _, r := range recs {
		r.Release()
	}
}

func TestTryReadBatch_CancelledContext(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 1, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(ctx, batch, alloc, identityTransform)
	// Either GetStream or context check will surface the error
	if err != nil {
		for _, r := range recs {
			r.Release()
		}
		assert.ErrorIs(t, err, context.Canceled)
		return
	}
	for _, r := range recs {
		r.Release()
	}
}

// --- readBatchRecords tests ---

func TestReadBatchRecords_SuccessFirstAttempt(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{5, 6})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 2, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, identityTransform, 3)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()

	assert.EqualValues(t, 2, recs[0].NumRows())
}

func TestReadBatchRecords_SuccessAfterRetries(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{7, 8, 9})
	defer rec.Release()

	goodData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})

	// First two calls fail, third succeeds
	batch := &mockBatch{numRows: 3, streams: []func(context.Context) (io.ReadCloser, error){
		streamError(fmt.Errorf("fail 1")),
		streamError(fmt.Errorf("fail 2")),
		streamFromBytes(goodData),
	}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, identityTransform, 3)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()

	assert.EqualValues(t, 3, recs[0].NumRows())
}

func TestReadBatchRecords_ExhaustsRetries(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	maxRetries := 2
	batch := &mockBatch{streams: []func(context.Context) (io.ReadCloser, error){
		streamError(fmt.Errorf("fail 1")),
		streamError(fmt.Errorf("fail 2")),
		streamError(fmt.Errorf("fail 3")),
	}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, identityTransform, maxRetries)
	require.Error(t, err)
	assert.Nil(t, recs)
	assert.Contains(t, err.Error(), "failed to read Arrow batch after 3 attempts")
	assert.Contains(t, err.Error(), "fail 3")
}

func TestReadBatchRecords_ZeroRetries(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	batch := &mockBatch{streams: []func(context.Context) (io.ReadCloser, error){
		streamError(fmt.Errorf("only chance")),
	}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, identityTransform, 0)
	require.Error(t, err)
	assert.Nil(t, recs)
	assert.Contains(t, err.Error(), "failed to read Arrow batch after 1 attempts")
}

func TestReadBatchRecords_CancelledContextSkipsRetries(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	batch := &mockBatch{streams: []func(context.Context) (io.ReadCloser, error){
		streamError(fmt.Errorf("should not reach")),
	}}

	recs, err := readBatchRecords(ctx, batch, alloc, identityTransform, 3)
	require.Error(t, err)
	assert.Nil(t, recs)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestReadBatchRecords_PartialRecordsReleasedOnRetry(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()

	// Build a good IPC stream for the success case (one record)
	goodRec := buildTestRecord(alloc, schema, []int64{100})
	defer goodRec.Release()
	goodData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{goodRec})

	// First attempt: IPC stream with two records. Transform succeeds on
	// the first record but fails on the second, simulating a partial-read
	// scenario where readBatchRecords must release the already-accumulated
	// records before retrying.
	partialRec1 := buildTestRecord(alloc, schema, []int64{42})
	defer partialRec1.Release()
	partialRec2 := buildTestRecord(alloc, schema, []int64{43})
	defer partialRec2.Release()
	failData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{partialRec1, partialRec2})

	transformCall := 0
	failOnSecondRecord := func(ctx context.Context, r arrow.RecordBatch) (arrow.RecordBatch, error) {
		transformCall++
		if transformCall == 2 {
			// Fail on the second record of the first attempt
			return nil, fmt.Errorf("mid-stream failure")
		}
		r.Retain()
		return r, nil
	}

	batch := &mockBatch{numRows: 1, streams: []func(context.Context) (io.ReadCloser, error){
		streamFromBytes(failData),
		streamFromBytes(goodData),
	}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, failOnSecondRecord, 3)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()

	// The allocator check in defer will catch any leaked memory from the
	// partial records of the failed first attempt.
	assert.EqualValues(t, 1, recs[0].NumRows())
}

// TestFixedToFloat64Transformer covers the driver code path that was changed by
// the NUMBER(38, 11) fix.
//
// getTransformer wires fixedToFloat64Transformer for every FIXED Snowflake column
// with useHighPrecision=false and scale != 0. The function chooses between two
// internal paths based on precision:
//
//   - precision <= 15: scale the int64 in place via compute.Divide (the unscaled
//     value safely fits in float64).
//   - precision >  15: widen to Decimal128 first, because the unscaled int64 can
//     exceed 2^53 and a direct int64->float64 safe cast would fail with
//     "integer value ... not in range: -9007199254740992 to 9007199254740992".
//
// Before the fix the precision>15 path was gated by an extra "precision < 19"
// upper bound, so NUMBER(p, s) columns with p >= 19 (e.g. NUMBER(38, 11)) fell
// through to compute.Divide and crashed on any unscaled value > 2^53. This test
// exercises both branches of the guard fixedToFloat64Transformer now owns.
func TestFixedToFloat64Transformer(t *testing.T) {
	cases := []struct {
		name          string
		precision     int32
		scale         int32
		unscaledValue int64
		want          float64
	}{
		{
			// Regression case: NUMBER(38, 11) with unscaled value > 2^53
			// (9007199254740992). Pre-fix this crashed; post-fix it goes through
			// the Decimal128 intermediate path.
			name:          "precision38_unscaledExceeds2Pow53",
			precision:     38,
			scale:         11,
			unscaledValue: 42135425651100000,
			want:          421354.256511,
		},
		{
			// Happy path that was always working: precision <= 15 uses the
			// in-place compute.Divide path. Included to guard against the
			// refactor accidentally rerouting small precisions.
			name:          "precision10_compactValue",
			precision:     10,
			scale:         2,
			unscaledValue: 12345,
			want:          123.45,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer alloc.AssertSize(t, 0)

			bldr := array.NewInt64Builder(alloc)
			defer bldr.Release()
			bldr.Append(tc.unscaledValue)
			int64Arr := bldr.NewInt64Array()
			defer int64Arr.Release()

			transformer := fixedToFloat64Transformer(tc.precision, tc.scale)
			out, err := transformer(context.Background(), int64Arr)
			require.NoError(t, err)
			defer out.Release()

			require.IsType(t, (*array.Float64)(nil), out)
			assert.InDelta(t, tc.want, out.(*array.Float64).Value(0), 1e-4)
		})
	}
}

func TestReadBatchRecords_RetriesAfterRowCountMismatch(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	shortRec := buildTestRecord(alloc, schema, []int64{1})
	defer shortRec.Release()
	goodRec := buildTestRecord(alloc, schema, []int64{1, 2})
	defer goodRec.Release()

	shortData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{shortRec})
	goodData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{goodRec})
	batch := &mockBatch{numRows: 2, streams: []func(context.Context) (io.ReadCloser, error){
		streamFromBytes(shortData),
		streamFromBytes(goodData),
	}}

	recs, err := readBatchRecords(context.Background(), batch, alloc, identityTransform, 1)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()

	assert.EqualValues(t, 2, recs[0].NumRows())
	assert.Equal(t, 2, batch.call)
}

func TestTryReadBatch_TruncatedStreamFailsRowCountValidation(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 3, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "batch stream row count mismatch: expected 3 rows, got 1")
	for _, r := range recs {
		r.Release()
	}
}

func TestStreamBatchToChannel_TruncatedStreamFailsRowCountValidation(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 3, streams: []func(context.Context) (io.ReadCloser, error){streamFromBytes(data)}}
	out := make(chan arrow.RecordBatch, 1)

	err := streamBatchToChannel(context.Background(), 2, batch, alloc, identityTransform, newBatchStreamTarget(2, batch, out, nil))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "batch[2] row count mismatch: expected 3 rows, got 1")

	select {
	case rec := <-out:
		assert.EqualValues(t, 1, rec.NumRows())
		rec.Release()
	default:
		require.FailNow(t, "streamBatchToChannel should have emitted the partial record before reporting the row count mismatch")
	}
}

func TestValidateRowCount_UsesNeutralMismatchMessage(t *testing.T) {
	err := validateRowCount("result set", 5, 3)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "result set row count mismatch: expected 5 rows, got 3")
	assert.NotContains(t, err.Error(), "ended early")
}

func TestStreamBatchToChannel_CancellationReturnsContextError(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	schemaOnly := truncateIPCStream(buildIPCBytes(alloc, schema, nil))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	streamCh := make(chan *contextEOFStream, 1)
	batch := &mockBatch{
		numRows: 0,
		streams: []func(context.Context) (io.ReadCloser, error){
			func(ctx context.Context) (io.ReadCloser, error) {
				stream := newContextEOFStream(ctx, schemaOnly)
				streamCh <- stream
				return stream, nil
			},
		},
	}

	out := make(chan arrow.RecordBatch, 1)
	errCh := make(chan error, 1)
	go func() {
		errCh <- streamBatchToChannel(ctx, 0, batch, alloc, identityTransform, newBatchStreamTarget(0, batch, out, nil))
	}()

	stream := <-streamCh
	<-stream.blocked
	cancel()

	err := <-errCh
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestReaderLateCancellationAfterLastRecordReturnsSuccess(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1})
	defer rec.Release()

	recordThenBlockAtEOF := truncateIPCStream(buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec}))

	ctx, cancel := context.WithCancel(context.Background())
	streamCh := make(chan *contextEOFStream, 1)
	chs := []chan arrow.RecordBatch{make(chan arrow.RecordBatch, 1)}
	rdr := &reader{
		refCount: 1,
		chs:      chs,
		cancelFn: cancel,
		done:     make(chan struct{}),
	}

	batch := &mockBatch{
		numRows: 1,
		streams: []func(context.Context) (io.ReadCloser, error){
			func(ctx context.Context) (io.ReadCloser, error) {
				stream := newContextEOFStream(ctx, recordThenBlockAtEOF)
				streamCh <- stream
				return stream, nil
			},
		},
	}

	go func() {
		rdr.setErr(streamBatchToChannel(ctx, 0, batch, alloc, identityTransform, newBatchStreamTarget(0, batch, chs[0], nil)))
		close(chs[0])
		close(rdr.done)
	}()

	stream := <-streamCh
	require.True(t, rdr.Next(), "reader should emit the completed batch before cancellation")
	assert.EqualValues(t, 1, rdr.RecordBatch().NumRows())

	<-stream.blocked
	cancel()

	nextCh := make(chan bool, 1)
	go func() {
		nextCh <- rdr.Next()
	}()

	select {
	case got := <-nextCh:
		assert.False(t, got)
	case <-time.After(200 * time.Millisecond):
		require.FailNow(t, "reader.Next should finish after late cancellation")
	}

	require.NoError(t, rdr.Err(), "late cancellation after the final batch should not surface as reader error")

	releaseDone := make(chan struct{})
	go func() {
		rdr.Release()
		close(releaseDone)
	}()
	select {
	case <-releaseDone:
	case <-time.After(200 * time.Millisecond):
		require.FailNow(t, "reader.Release should not block after late cancellation")
	}
}

func TestReaderStopsAfterEarlierBatchFailureDespiteLaterPrefetch(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	failingRec := buildTestRecord(alloc, schema, []int64{1})
	defer failingRec.Release()
	laterRec := buildTestRecord(alloc, schema, []int64{2})
	defer laterRec.Release()

	shortData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{failingRec})
	laterData := buildIPCBytes(alloc, schema, []arrow.RecordBatch{laterRec})

	ctx, cancel := context.WithCancel(context.Background())
	chs := []chan arrow.RecordBatch{
		make(chan arrow.RecordBatch, 1),
		make(chan arrow.RecordBatch, 1),
	}
	rdr := &reader{
		refCount: 1,
		chs:      chs,
		cancelFn: cancel,
		done:     make(chan struct{}),
	}

	failingBatch := &mockBatch{
		numRows: 2,
		streams: []func(context.Context) (io.ReadCloser, error){
			streamFromBytes(shortData),
		},
	}
	laterBatch := &mockBatch{
		numRows: 1,
		streams: []func(context.Context) (io.ReadCloser, error){
			streamFromBytes(laterData),
		},
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		err := streamBatchToChannel(ctx, 1, laterBatch, alloc, identityTransform, newBatchStreamTarget(1, laterBatch, chs[1], nil))
		rdr.setErr(err)
		close(chs[1])
	}()

	require.Eventually(t, func() bool {
		return len(chs[1]) == 1
	}, 200*time.Millisecond, 10*time.Millisecond, "later batch should prefetch before the earlier batch fails")

	go func() {
		defer wg.Done()
		err := streamBatchToChannel(ctx, 0, failingBatch, alloc, identityTransform, newBatchStreamTarget(0, failingBatch, chs[0], nil))
		rdr.setErr(err)
		close(chs[0])
	}()

	go func() {
		wg.Wait()
		close(rdr.done)
	}()

	require.True(t, rdr.Next(), "reader should emit the partial record from the failing batch first")
	assert.EqualValues(t, 1, rdr.RecordBatch().NumRows())

	assert.False(t, rdr.Next(), "reader should stop before yielding rows from later prefetched batches")
	require.Error(t, rdr.Err())
	assert.Contains(t, rdr.Err().Error(), "batch[0] row count mismatch: expected 2 rows, got 1")
	assert.Len(t, chs[1], 1, "later prefetched rows should remain queued for release, not be returned by Next")

	releaseDone := make(chan struct{})
	go func() {
		rdr.Release()
		close(releaseDone)
	}()
	select {
	case <-releaseDone:
	case <-time.After(200 * time.Millisecond):
		require.FailNow(t, "reader.Release should not block after suppressing later prefetched rows")
	}
}

func TestReaderCancellationSetsErrBeforeNextAndReleaseReturns(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	schemaOnly := truncateIPCStream(buildIPCBytes(alloc, schema, nil))

	ctx, cancel := context.WithCancel(context.Background())
	streamCh := make(chan *contextEOFStream, 1)
	chs := []chan arrow.RecordBatch{make(chan arrow.RecordBatch)}
	rdr := &reader{
		refCount: 1,
		chs:      chs,
		cancelFn: cancel,
		done:     make(chan struct{}),
	}

	batch := &mockBatch{
		numRows: 0,
		streams: []func(context.Context) (io.ReadCloser, error){
			func(ctx context.Context) (io.ReadCloser, error) {
				stream := newContextEOFStream(ctx, schemaOnly)
				streamCh <- stream
				return stream, nil
			},
		},
	}

	go func() {
		rdr.setErr(streamBatchToChannel(ctx, 0, batch, alloc, identityTransform, newBatchStreamTarget(0, batch, chs[0], nil)))
		close(chs[0])
		close(rdr.done)
	}()

	nextCh := make(chan bool, 1)
	go func() {
		nextCh <- rdr.Next()
	}()

	stream := <-streamCh
	<-stream.blocked
	cancel()

	select {
	case got := <-nextCh:
		assert.False(t, got)
	case <-time.After(200 * time.Millisecond):
		require.FailNow(t, "reader.Next should not block after cancellation")
	}
	require.Error(t, rdr.Err())
	assert.ErrorIs(t, rdr.Err(), context.Canceled)

	releaseDone := make(chan struct{})
	go func() {
		rdr.Release()
		close(releaseDone)
	}()
	select {
	case <-releaseDone:
	case <-time.After(200 * time.Millisecond):
		require.FailNow(t, "reader.Release should not block after cancellation")
	}
}

// shortReadStream delivers data in chunks no larger than chunkSize per Read.
// If errAtOffset > 0, it returns injectedErr once that many bytes have been
// delivered, simulating mid-frame truncation (gosnowflake#1781).
type shortReadStream struct {
	data        []byte
	pos         int
	chunkSize   int
	errAtOffset int
	injectedErr error
}

func (s *shortReadStream) Read(p []byte) (int, error) {
	if s.errAtOffset > 0 && s.pos >= s.errAtOffset {
		return 0, s.injectedErr
	}
	if s.pos >= len(s.data) {
		return 0, io.EOF
	}
	n := min(len(p), s.chunkSize)
	remaining := len(s.data) - s.pos
	if n > remaining {
		n = remaining
	}
	if s.errAtOffset > 0 && s.pos+n > s.errAtOffset {
		n = s.errAtOffset - s.pos
	}
	copy(p, s.data[s.pos:s.pos+n])
	s.pos += n
	return n, nil
}

func (s *shortReadStream) Close() error { return nil }

func streamShortReads(data []byte, chunkSize int) func(context.Context) (io.ReadCloser, error) {
	return func(context.Context) (io.ReadCloser, error) {
		return &shortReadStream{data: data, chunkSize: chunkSize}, nil
	}
}

func streamShortReadsThenError(data []byte, chunkSize, errAtOffset int, err error) func(context.Context) (io.ReadCloser, error) {
	return func(context.Context) (io.ReadCloser, error) {
		return &shortReadStream{
			data:        data,
			chunkSize:   chunkSize,
			errAtOffset: errAtOffset,
			injectedErr: err,
		}, nil
	}
}

// Sanity check that legal short reads alone (n < len(p), nil err) don't break
// IPC decoding — the failure in gosnowflake#1781 requires a mid-frame error.
func TestTryReadBatch_ShortReadsSucceed(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := testSchema()
	rec := buildTestRecord(alloc, schema, []int64{1, 2, 3, 4, 5})
	defer rec.Release()

	data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
	batch := &mockBatch{numRows: 5, streams: []func(context.Context) (io.ReadCloser, error){
		streamShortReads(data, 1),
	}}

	recs, err := tryReadBatch(context.Background(), batch, alloc, identityTransform)
	require.NoError(t, err)
	require.Len(t, recs, 1)
	defer recs[0].Release()
	assert.EqualValues(t, 5, recs[0].NumRows())
}

// Reproduces gosnowflake#1781 and validates the streamRetryEnabled flag:
// with retries disabled (matching the production "no retry" branch in
// newRecordReader) the mid-frame IPC error surfaces; with retries enabled
// the same broken first stream is recovered by a second attempt.
func TestStreamRetryEnabled_RecoversFromShortReadMidFrameError(t *testing.T) {
	schema := testSchema()

	cases := []struct {
		name               string
		streamRetryEnabled bool
		expectErr          bool
	}{
		{name: "disabled_surfacesIPCError", streamRetryEnabled: false, expectErr: true},
		{name: "enabled_recovers", streamRetryEnabled: true, expectErr: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
			defer alloc.AssertSize(t, 0)

			rec := buildTestRecord(alloc, schema, []int64{11, 22, 33, 44})
			defer rec.Release()

			data := buildIPCBytes(alloc, schema, []arrow.RecordBatch{rec})
			require.Greater(t, len(data), 32)
			truncationOffset := len(data) - 8

			batch := &mockBatch{numRows: 4, streams: []func(context.Context) (io.ReadCloser, error){
				streamShortReadsThenError(data, 16, truncationOffset, io.ErrUnexpectedEOF),
				streamShortReads(data, 16),
			}}

			// Mirror the production dispatch in newRecordReader.
			ctx := context.Background()
			var recs []arrow.RecordBatch
			var err error
			if tc.streamRetryEnabled {
				recs, err = readBatchRecords(ctx, batch, alloc, identityTransform, defaultStreamMaxRetries)
			} else {
				out := make(chan arrow.RecordBatch, 4)
				target := newBatchStreamTarget(0, batch, out, nil)
				err = streamBatchToChannel(ctx, 0, batch, alloc, identityTransform, target)
				close(out)
				for r := range out {
					recs = append(recs, r)
				}
			}

			if tc.expectErr {
				require.Error(t, err, "expected IPC error to surface without retry")
				msg := err.Error()
				assert.True(t,
					strings.Contains(msg, "could not read message body") ||
						strings.Contains(msg, "unexpected EOF") ||
						strings.Contains(msg, "row count mismatch"),
					"expected IPC body-read failure, got: %s", msg,
				)
				for _, r := range recs {
					r.Release()
				}
				return
			}

			require.NoError(t, err, "retry should recover from mid-frame truncation")
			require.Len(t, recs, 1)
			defer recs[0].Release()
			assert.EqualValues(t, 4, recs[0].NumRows())
			assert.Equal(t, 2, batch.call, "retry should have been invoked")
		})
	}
}

// POINT(1 2) with SRID 4326, as EWKB and as the expected plain WKB.
var (
	ewkbPointLE = []byte{
		0x01, 0x01, 0x00, 0x00, 0x20, // little endian, Point | SRID flag
		0xE6, 0x10, 0x00, 0x00, // SRID 4326
		0, 0, 0, 0, 0, 0, 0xF0, 0x3F, // x = 1
		0, 0, 0, 0, 0, 0, 0x00, 0x40, // y = 2
	}
	wkbPointLE = []byte{
		0x01, 0x01, 0x00, 0x00, 0x00,
		0, 0, 0, 0, 0, 0, 0xF0, 0x3F,
		0, 0, 0, 0, 0, 0, 0x00, 0x40,
	}
	ewkbPointBE = []byte{
		0x00, 0x20, 0x00, 0x00, 0x01, // big endian, SRID flag | Point
		0x00, 0x00, 0x10, 0xE6,
		0x3F, 0xF0, 0, 0, 0, 0, 0, 0,
		0x40, 0x00, 0, 0, 0, 0, 0, 0,
	}
	wkbPointBE = []byte{
		0x00, 0x00, 0x00, 0x00, 0x01,
		0x3F, 0xF0, 0, 0, 0, 0, 0, 0,
		0x40, 0x00, 0, 0, 0, 0, 0, 0,
	}
)

func TestAppendStrippedEWKBSRID(t *testing.T) {
	tests := []struct {
		name    string
		in, out []byte
	}{
		{"little endian with SRID", ewkbPointLE, wkbPointLE},
		{"big endian with SRID", ewkbPointBE, wkbPointBE},
		{"plain WKB is unchanged", wkbPointLE, wkbPointLE},
		{"short input is unchanged", []byte{0x01, 0x02}, []byte{0x01, 0x02}},
		{"unknown byte order is unchanged", append([]byte{0x07}, ewkbPointLE[1:]...), append([]byte{0x07}, ewkbPointLE[1:]...)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.out, appendStrippedEWKBSRID(nil, tt.in))
			// Appends after existing content rather than overwriting it.
			assert.Equal(t, append([]byte("x"), tt.out...), appendStrippedEWKBSRID([]byte("x"), tt.in))
		})
	}
}

func TestStripEWKBSRIDColumnUsesContextAllocator(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)

	bldr := array.NewBinaryBuilder(mem, arrow.BinaryTypes.Binary)
	bldr.Append(ewkbPointLE)
	bldr.AppendNull()
	bldr.Append(ewkbPointBE)
	bldr.Append(wkbPointLE)
	in := bldr.NewArray()
	bldr.Release()
	defer in.Release()
	inBytes := mem.CurrentAlloc()

	out, err := stripEWKBSRIDColumn(compute.WithAllocator(context.Background(), mem), in)
	require.NoError(t, err)
	assert.Greater(t, mem.CurrentAlloc(), inBytes, "output was not allocated from the context allocator")

	got := out.(*array.Binary)
	require.Equal(t, 4, got.Len())
	assert.Equal(t, wkbPointLE, got.Value(0))
	assert.True(t, got.IsNull(1))
	assert.Equal(t, wkbPointBE, got.Value(2))
	assert.Equal(t, wkbPointLE, got.Value(3))
	out.Release()
}

func BenchmarkStripEWKBSRIDColumn(b *testing.B) {
	bldr := array.NewBinaryBuilder(memory.DefaultAllocator, arrow.BinaryTypes.Binary)
	for range 1 << 16 {
		bldr.Append(ewkbPointLE)
	}
	in := bldr.NewArray()
	bldr.Release()
	defer in.Release()

	ctx := compute.WithAllocator(context.Background(), memory.DefaultAllocator)
	b.ReportAllocs()
	for b.Loop() {
		out, err := stripEWKBSRIDColumn(ctx, in)
		if err != nil {
			b.Fatal(err)
		}
		out.Release()
	}
}

// jsonRow builds an expected decoded row; "<null>" stands for a JSON null.
func jsonRow(vals ...string) []*string {
	row := make([]*string, len(vals))
	for i, v := range vals {
		if v != "<null>" {
			row[i] = &v
		}
	}
	return row
}

func TestDecodeJSONChunkRows(t *testing.T) {
	chunks := map[string]struct {
		data string
		want [][]*string
	}{
		"single row":               {`["a","b"]`, [][]*string{jsonRow("a", "b")}},
		"nulls":                    {`[null,"1"],["2",null]`, [][]*string{jsonRow("<null>", "1"), jsonRow("2", "<null>")}},
		"escapes and unicode":      {`["a\"b\\c","é\n\u00e9"],["[,]","{\"k\":1}"]`, [][]*string{jsonRow(`a"b\c`, "é\né"), jsonRow("[,]", `{"k":1}`)}},
		"invalid UTF-8":            {"[\"a\xffb\",\"\\ud800\"]", [][]*string{jsonRow("a\uFFFDb", "\uFFFD")}},
		"whitespace between rows":  {"[\"1\"],\n[\"2\"],\t [\"3\"]", [][]*string{jsonRow("1"), jsonRow("2"), jsonRow("3")}},
		"fewer rows than in chunk": {`["1"],["2"],["3"]`, [][]*string{jsonRow("1"), jsonRow("2")}},
	}
	for name, tc := range chunks {
		t.Run(name, func(t *testing.T) {
			got := make([][]*string, len(tc.want))
			require.NoError(t, decodeJSONChunkRows([]byte(tc.data), len(tc.want[0]), got))
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestDecodeJSONChunkRowsErrors(t *testing.T) {
	for name, data := range map[string]string{
		"too few rows":         `["1","2"]`,
		"not a list":           `{"a":"1"}`,
		"number value":         `["1","2"],["3",4]`,
		"too few values":       `["1","2"],["3"]`,
		"too many values":      `["1","2"],["3","4","5"]`,
		"truncated":            `["1","2"],["3","4"`,
		"truncated mid-string": `["1","2"],["3","4`,
	} {
		t.Run(name, func(t *testing.T) {
			err := decodeJSONChunkRows([]byte(data), 2, make([][]*string, 2))
			assert.ErrorContains(t, err, "JSON decode failed")
		})
	}
}

func BenchmarkDecodeJSONChunkRows(b *testing.B) {
	const rows, cols = 50_000, 8
	var sb strings.Builder
	for r := range rows {
		if r > 0 {
			sb.WriteByte(',')
		}
		sb.WriteByte('[')
		for c := range cols {
			if c > 0 {
				sb.WriteByte(',')
			}
			if (r+c)%7 == 0 {
				sb.WriteString("null")
			} else {
				fmt.Fprintf(&sb, "%q", fmt.Sprintf("row %d col %d", r, c))
			}
		}
		sb.WriteByte(']')
	}
	data := []byte(sb.String())
	dst := make([][]*string, rows)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	for b.Loop() {
		if err := decodeJSONChunkRows(data, cols, dst); err != nil {
			b.Fatal(err)
		}
	}
}

// timestampColumn builds a column in one of the layouts Snowflake sends
// timestamps in: int64 when every value has one part, otherwise a struct of
// epoch with fraction and/or timezone. Each row is one value's parts; a nil
// row is null.
func timestampColumn(mem memory.Allocator, withOffset bool, rows ...[]int64) arrow.Array {
	parts := 0
	for _, r := range rows {
		parts = max(parts, len(r))
	}
	if parts == 1 {
		b := array.NewInt64Builder(mem)
		defer b.Release()
		for _, r := range rows {
			if r == nil {
				b.AppendNull()
			} else {
				b.Append(r[0])
			}
		}
		return b.NewArray()
	}
	fields := []arrow.Field{{Name: "epoch", Type: arrow.PrimitiveTypes.Int64}}
	if parts == 3 || !withOffset {
		fields = append(fields, arrow.Field{Name: "fraction", Type: arrow.PrimitiveTypes.Int32})
	}
	if withOffset {
		fields = append(fields, arrow.Field{Name: "timezone", Type: arrow.PrimitiveTypes.Int32})
	}
	sb := array.NewStructBuilder(mem, arrow.StructOf(fields...))
	defer sb.Release()
	for _, r := range rows {
		if r == nil {
			sb.AppendNull()
			continue
		}
		sb.Append(true)
		sb.FieldBuilder(0).(*array.Int64Builder).Append(r[0])
		for f := 1; f < len(r); f++ {
			sb.FieldBuilder(f).(*array.Int32Builder).Append(int32(r[f]))
		}
	}
	return sb.NewArray()
}

func TestTimestampTransformer(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	const overflow = "overflow"
	modes := []MaxTimestampPrecision{Nanoseconds, NanosecondsNoOverflow, Microseconds}
	// want holds the result for Nanoseconds, NanosecondsNoOverflow and
	// Microseconds, in the unit getArrowTimeUnit picks for each.
	cases := []struct {
		name       string
		withOffset bool
		scale      int64
		in         []int64
		want       [3]any
	}{
		{"scale 0", false, 0, []int64{1717200000}, [3]any{1717200000, 1717200000, 1717200000}},
		{"scale 0 before epoch", false, 0, []int64{-1}, [3]any{-1, -1, -1}},
		{"scale 0 beyond nanoseconds", false, 0, []int64{9223372037}, [3]any{9223372037, 9223372037, 9223372037}},
		{"scale 3", false, 3, []int64{1717200000123}, [3]any{1717200000123, 1717200000123, 1717200000123}},
		{"scale 3 before epoch", false, 3, []int64{-1}, [3]any{-1, -1, -1}},
		{"scale 7", false, 7, []int64{17172000001234567}, [3]any{1717200000123456700, 1717200000123456700, 1717200000123456}},
		{"scale 7 before epoch", false, 7, []int64{-1}, [3]any{-100, -100, -1}},
		{"scale 7 before epoch, whole microseconds", false, 7, []int64{-10000001}, [3]any{-1000000100, -1000000100, -1000001}},
		{"scale 7 max", false, 7, []int64{math.MaxInt64}, [3]any{-100, overflow, 922337203685477580}},
		{"scale 7 min", false, 7, []int64{math.MinInt64}, [3]any{0, overflow, -922337203685477581}},
		{"scale 8", false, 8, []int64{1717200000, 123456780}, [3]any{1717200000123456780, 1717200000123456780, 1717200000123456}},
		{"scale 8 before epoch", false, 8, []int64{-1, 500000000}, [3]any{-500000000, -500000000, -500000}},
		{"scale 9", false, 9, []int64{1717200000, 123456789}, [3]any{1717200000123456789, 1717200000123456789, 1717200000123456}},
		{"scale 9 largest nanoseconds", false, 9, []int64{9223372036, 854775807}, [3]any{int64(math.MaxInt64), int64(math.MaxInt64), 9223372036854775}},
		{"scale 9 past largest nanoseconds", false, 9, []int64{9223372036, 854775808}, [3]any{int64(math.MinInt64), overflow, 9223372036854775}},
		{"scale 9 before smallest nanoseconds", false, 9, []int64{-9223372037, 145224191}, [3]any{int64(math.MaxInt64), overflow, -9223372036854776}},
		{"scale 9 year 9999", false, 9, []int64{253402300799, 999999999}, [3]any{-4852116231933722625, overflow, 253402300799999999}},
		{"TZ scale 3", true, 3, []int64{1717200000123, 1500}, [3]any{1717200000123, 1717200000123, 1717200000123}},
		{"TZ scale 3 before epoch", true, 3, []int64{-1, 1380}, [3]any{-1, -1, -1}},
		{"TZ scale 9", true, 9, []int64{1717200000, 5, 1500}, [3]any{1717200000000000005, 1717200000000000005, 1717200000000000}},
		{"TZ scale 9 before epoch", true, 9, []int64{-1, 999999999, 1440}, [3]any{-1, -1, -1}},
		{"TZ scale 9 beyond nanoseconds", true, 9, []int64{9223372037, 0, 1440}, [3]any{-9223372036709551616, overflow, 9223372037000000}},
	}
	for _, tc := range cases {
		for i, mode := range modes {
			t.Run(fmt.Sprintf("%s/%v", tc.name, mode), func(t *testing.T) {
				tz := ""
				if tc.withOffset {
					tz = "UTC"
				}
				dt := &arrow.TimestampType{Unit: getArrowTimeUnit(tc.scale, mode), TimeZone: tz}
				in := timestampColumn(mem, tc.withOffset, tc.in)
				defer in.Release()
				got, err := timestampTransformer(tc.withOffset, dt, tc.scale, getArrowTimeUnit(tc.scale, Nanoseconds), mode, nil)(ctx, in)
				if tc.want[i] == overflow {
					assert.ErrorContains(t, err, "timestamp")
					return
				}
				require.NoError(t, err)
				defer got.Release()
				assert.True(t, arrow.TypeEqual(dt, got.DataType()), "type %v", got.DataType())
				var want int64
				switch w := tc.want[i].(type) {
				case int:
					want = int64(w)
				case int64:
					want = w
				}
				assert.Equal(t, []arrow.Timestamp{arrow.Timestamp(want)}, got.(*array.Timestamp).TimestampValues())
			})
		}
	}
}

func TestTimestampTransformerNullsAndSlices(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	for _, tc := range []struct {
		name       string
		withOffset bool
		scale      int64
		rows       [][]int64
		want       []arrow.Timestamp // nanoseconds
	}{
		{"int64", false, 7, [][]int64{{1}, nil, {-1}, {17172000001234567}, nil}, []arrow.Timestamp{100, 0, -100, 1717200000123456700, 0}},
		{"epoch and fraction", false, 9, [][]int64{{1, 2}, nil, {-1, 500000000}, {1717200000, 123456789}, nil}, []arrow.Timestamp{1000000002, 0, -500000000, 1717200000123456789, 0}},
		{"epoch, fraction and timezone", true, 9, [][]int64{{1, 2, 1440}, nil, {-1, 500000000, 1500}, {1717200000, 123456789, 1380}, nil}, []arrow.Timestamp{1000000002, 0, -500000000, 1717200000123456789, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			full := timestampColumn(mem, tc.withOffset, tc.rows...)
			defer full.Release()
			sliced := array.NewSlice(full, 1, 4)
			defer sliced.Release()
			dt := &arrow.TimestampType{Unit: arrow.Nanosecond}
			if tc.withOffset {
				dt.TimeZone = "UTC"
			}
			fn := timestampTransformer(tc.withOffset, dt, tc.scale, arrow.Nanosecond, Nanoseconds, nil)
			for _, c := range []struct {
				in   arrow.Array
				want []arrow.Timestamp
			}{{full, tc.want}, {sliced, tc.want[1:4]}} {
				got, err := fn(ctx, c.in)
				require.NoError(t, err)
				ts := got.(*array.Timestamp)
				require.Equal(t, c.in.Len(), ts.Len())
				for i := range ts.Len() {
					assert.Equal(t, c.in.IsNull(i), ts.IsNull(i), "row %d", i)
					if !ts.IsNull(i) {
						assert.Equal(t, c.want[i], ts.Value(i), "row %d", i)
					}
				}
				got.Release()
			}
		})
	}
}

// Values that are already in the output unit share the input's buffer.
func TestTimestampTransformerPassThrough(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	sameBuffer := func(t *testing.T, got arrow.Array, values *memory.Buffer) {
		t.Helper()
		assert.Same(t, &values.Bytes()[0], &got.Data().Buffers()[1].Bytes()[0], "values were copied")
	}
	values := func(a arrow.Array) []arrow.Timestamp {
		ts := a.(*array.Timestamp)
		out := make([]arrow.Timestamp, ts.Len())
		for i := range out {
			if ts.IsValid(i) {
				out[i] = ts.Value(i)
			}
		}
		return out
	}
	ms := &arrow.TimestampType{Unit: arrow.Millisecond}
	msUTC := &arrow.TimestampType{Unit: arrow.Millisecond, TimeZone: "UTC"}

	t.Run("int64", func(t *testing.T) {
		in := timestampColumn(mem, false, []int64{1717200000123}, nil, []int64{-1})
		defer in.Release()
		sliced := array.NewSlice(in, 1, 3)
		defer sliced.Release()
		fn := timestampTransformer(false, ms, 3, arrow.Millisecond, Nanoseconds, nil)
		for _, c := range []struct {
			in   arrow.Array
			want []arrow.Timestamp
		}{{in, []arrow.Timestamp{1717200000123, 0, -1}}, {sliced, []arrow.Timestamp{0, -1}}} {
			got, err := fn(ctx, c.in)
			require.NoError(t, err)
			assert.True(t, arrow.TypeEqual(ms, got.DataType()))
			assert.Equal(t, c.in.NullN(), got.NullN())
			assert.Equal(t, c.want, values(got))
			sameBuffer(t, got, in.Data().Buffers()[1])
			got.Release()
		}
	})

	t.Run("outlives input", func(t *testing.T) {
		in := timestampColumn(mem, false, []int64{1}, []int64{2}, []int64{3})
		got, err := timestampTransformer(false, &arrow.TimestampType{Unit: arrow.Second}, 0, arrow.Second, Nanoseconds, nil)(ctx, in)
		require.NoError(t, err)
		in.Release()
		assert.Equal(t, []arrow.Timestamp{1, 2, 3}, got.(*array.Timestamp).TimestampValues())
		got.Release()
	})

	t.Run("out of range value in a null slot", func(t *testing.T) {
		ns := &arrow.TimestampType{Unit: arrow.Nanosecond}
		ib := array.NewInt64Builder(mem)
		ib.AppendValues([]int64{0, math.MinInt64, 5}, []bool{true, false, true})
		in := ib.NewArray()
		ib.Release()
		defer in.Release()
		got, err := timestampTransformer(false, ns, 9, arrow.Nanosecond, NanosecondsNoOverflow, nil)(ctx, in)
		require.NoError(t, err)
		assert.Equal(t, []arrow.Timestamp{0, 0, 5}, values(got))
		sameBuffer(t, got, in.Data().Buffers()[1])
		got.Release()

		ib = array.NewInt64Builder(mem)
		ib.AppendValues([]int64{0, math.MinInt64}, nil)
		bad := ib.NewArray()
		ib.Release()
		defer bad.Release()
		_, err = timestampTransformer(false, ns, 9, arrow.Nanosecond, NanosecondsNoOverflow, nil)(ctx, bad)
		assert.ErrorContains(t, err, "overflows when converted to nanoseconds")
	})

	t.Run("epoch and timezone", func(t *testing.T) {
		in := timestampColumn(mem, true, []int64{1717200000123, 1500}, nil, []int64{-1, 1380})
		defer in.Release()
		sliced := array.NewSlice(in, 1, 3)
		defer sliced.Release()
		fn := timestampTransformer(true, msUTC, 3, arrow.Millisecond, Nanoseconds, nil)
		epochs := in.(*array.Struct).Field(0).Data().Buffers()[1]
		for _, c := range []struct {
			in   arrow.Array
			want []arrow.Timestamp
		}{{in, []arrow.Timestamp{1717200000123, 0, -1}}, {sliced, []arrow.Timestamp{0, -1}}} {
			got, err := fn(ctx, c.in)
			require.NoError(t, err)
			assert.True(t, arrow.TypeEqual(msUTC, got.DataType()))
			assert.Equal(t, c.want, values(got))
			sameBuffer(t, got, epochs)
			got.Release()
		}
	})

	t.Run("struct children offset from the struct", func(t *testing.T) {
		// The struct's validity starts at 0 but its children's values
		// don't, so they can't be shared.
		long := timestampColumn(mem, false, []int64{10}, []int64{11}, []int64{12}, []int64{13})
		defer long.Release()
		epochs := array.NewSlice(long, 1, 4)
		defer epochs.Release()
		zb := array.NewInt32Builder(mem)
		zb.AppendValues([]int32{1500, 1500, 1500, 1500}, nil)
		zonesFull := zb.NewArray()
		zb.Release()
		defer zonesFull.Release()
		zones := array.NewSlice(zonesFull, 1, 4)
		defer zones.Release()
		in, err := array.NewStructArrayWithFields([]arrow.Array{epochs, zones}, []arrow.Field{
			{Name: "epoch", Type: arrow.PrimitiveTypes.Int64},
			{Name: "timezone", Type: arrow.PrimitiveTypes.Int32},
		})
		require.NoError(t, err)
		defer in.Release()
		require.NotEqual(t, in.Data().Offset(), in.Field(0).Data().Offset())

		got, err := timestampTransformer(true, msUTC, 3, arrow.Millisecond, Nanoseconds, nil)(ctx, in)
		require.NoError(t, err)
		assert.Equal(t, []arrow.Timestamp{11, 12, 13}, values(got))
		assert.NotSame(t, &long.Data().Buffers()[1].Bytes()[0], &got.Data().Buffers()[1].Bytes()[0])
		got.Release()
	})
}

// BenchmarkTimestampTransformer uses the representations Snowflake sends:
// TIMESTAMP_NTZ/LTZ are int64 up to scale 7 and (epoch, fraction) above it, and
// TIMESTAMP_TZ is (epoch, timezone) up to scale 3 and (epoch, fraction,
// timezone) above it.
func BenchmarkTimestampTransformer(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	const n = 1 << 16
	const epochSeconds = 1717200000
	compact := func(scale int64) arrow.Array {
		pow := int64(math.Pow10(int(scale)))
		ib := array.NewInt64Builder(memory.DefaultAllocator)
		defer ib.Release()
		for i := range n {
			if i%17 == 0 {
				ib.AppendNull()
				continue
			}
			ib.Append((epochSeconds+r.Int64N(1e9)-5e8)*pow + r.Int64N(pow))
		}
		return ib.NewArray()
	}
	structOf := func(fields ...string) arrow.Array {
		var fs []arrow.Field
		for _, f := range fields {
			typ := arrow.DataType(arrow.PrimitiveTypes.Int32)
			if f == "epoch" {
				typ = arrow.PrimitiveTypes.Int64
			}
			fs = append(fs, arrow.Field{Name: f, Type: typ})
		}
		sb := array.NewStructBuilder(memory.DefaultAllocator, arrow.StructOf(fs...))
		defer sb.Release()
		for i := range n {
			if i%17 == 0 {
				sb.AppendNull()
				continue
			}
			sb.Append(true)
			for f, name := range fields {
				switch name {
				case "epoch":
					if len(fields) == 2 && fields[1] == "timezone" {
						// Compact TIMESTAMP_TZ at scale 3.
						sb.FieldBuilder(f).(*array.Int64Builder).Append((epochSeconds+r.Int64N(1e9)-5e8)*1000 + r.Int64N(1000))
					} else {
						sb.FieldBuilder(f).(*array.Int64Builder).Append(epochSeconds + r.Int64N(1e9) - 5e8)
					}
				case "fraction":
					sb.FieldBuilder(f).(*array.Int32Builder).Append(r.Int32N(1e9))
				case "timezone":
					sb.FieldBuilder(f).(*array.Int32Builder).Append(1440 + 60)
				}
			}
		}
		return sb.NewArray()
	}

	ntz6, ntz7 := compact(6), compact(7)
	ntz9 := structOf("epoch", "fraction")
	tz3 := structOf("epoch", "timezone")
	tz9 := structOf("epoch", "fraction", "timezone")
	for _, a := range []arrow.Array{ntz6, ntz7, ntz9, tz3, tz9} {
		defer a.Release()
	}

	ctx := compute.WithAllocator(context.Background(), memory.DefaultAllocator)
	cases := []struct {
		name       string
		in         arrow.Array
		scale      int64
		mode       MaxTimestampPrecision
		withOffset bool
	}{
		// Already in the output unit: passed through.
		{"NTZ_6", ntz6, 6, Nanoseconds, false},
		{"TZ_3", tz3, 3, Nanoseconds, true},
		// Rescaled per value.
		{"NTZ_7", ntz7, 7, Nanoseconds, false},
		{"NTZ_9", ntz9, 9, Nanoseconds, false},
		{"NTZ_9_Micro", ntz9, 9, Microseconds, false},
		{"TZ_9", tz9, 9, Nanoseconds, true},
	}
	for _, c := range cases {
		tz := ""
		if c.withOffset {
			tz = "UTC"
		}
		dt := &arrow.TimestampType{Unit: getArrowTimeUnit(c.scale, c.mode), TimeZone: tz}
		orig := getArrowTimeUnit(c.scale, Nanoseconds)
		fn := timestampTransformer(c.withOffset, dt, c.scale, orig, c.mode, nil)
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				out, err := fn(ctx, c.in)
				if err != nil {
					b.Fatal(err)
				}
				out.Release()
			}
		})
	}
}
