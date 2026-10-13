// Copyright (c) 2026 ADBC Drivers Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package snowflake

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// legacyTimestampTransformer is the previous per-value implementation of the
// TIMESTAMP_* transformers in getTransformer, kept verbatim as an oracle.
func legacyTimestampTransformer(typ string, dt *arrow.TimestampType, scale int64, originalArrowUnit arrow.TimeUnit, maxTimestampPrecision MaxTimestampPrecision, loc *time.Location) colTransformer {
	scaleDivisor := int64(math.Pow10(int(scale)))
	fractionMultiplier := int64(math.Pow10(9 - int(scale)))
	switch typ {
	case "TIMESTAMP_NTZ":
		return func(ctx context.Context, a arrow.Array) (arrow.Array, error) {
			tb := array.NewTimestampBuilder(compute.GetAllocator(ctx), dt)
			defer tb.Release()
			if a.DataType().ID() != arrow.STRUCT {
				for i, value := range a.(*array.Int64).Int64Values() {
					if a.IsNull(i) {
						tb.AppendNull()
						continue
					}
					v, err := getArrowTimestampFromTime(scaledTimestampToTime(value, scaleDivisor, fractionMultiplier), dt.TimeUnit(), originalArrowUnit, maxTimestampPrecision)
					if err != nil {
						return nil, err
					}
					tb.Append(v)
				}
				return tb.NewArray(), nil
			}
			structData := a.(*array.Struct)
			epoch := structData.Field(0).(*array.Int64).Int64Values()
			fraction := structData.Field(1).(*array.Int32).Int32Values()
			for i := range a.Len() {
				if a.IsNull(i) {
					tb.AppendNull()
					continue
				}
				v, err := getArrowTimestampFromTime(time.Unix(epoch[i], int64(fraction[i])), dt.TimeUnit(), originalArrowUnit, maxTimestampPrecision)
				if err != nil {
					return nil, err
				}
				tb.Append(v)
			}
			return tb.NewArray(), nil
		}
	case "TIMESTAMP_LTZ":
		return func(ctx context.Context, a arrow.Array) (arrow.Array, error) {
			tb := array.NewTimestampBuilder(compute.GetAllocator(ctx), dt)
			defer tb.Release()
			if a.DataType().ID() == arrow.STRUCT {
				structData := a.(*array.Struct)
				epoch := structData.Field(0).(*array.Int64).Int64Values()
				fraction := structData.Field(1).(*array.Int32).Int32Values()
				for i := range a.Len() {
					if a.IsNull(i) {
						tb.AppendNull()
						continue
					}
					v, err := getArrowTimestampFromTime(time.Unix(epoch[i], int64(fraction[i])), dt.TimeUnit(), originalArrowUnit, maxTimestampPrecision)
					if err != nil {
						return nil, err
					}
					tb.Append(v)
				}
			} else {
				for i, value := range a.(*array.Int64).Int64Values() {
					if a.IsNull(i) {
						tb.AppendNull()
						continue
					}
					t := scaledTimestampToTime(value, scaleDivisor, fractionMultiplier).In(loc)
					v, err := getArrowTimestampFromTime(t, dt.TimeUnit(), originalArrowUnit, maxTimestampPrecision)
					if err != nil {
						return nil, err
					}
					tb.Append(v)
				}
			}
			return tb.NewArray(), nil
		}
	default: // TIMESTAMP_TZ
		return func(ctx context.Context, a arrow.Array) (arrow.Array, error) {
			tb := array.NewTimestampBuilder(compute.GetAllocator(ctx), dt)
			defer tb.Release()
			structData := a.(*array.Struct)
			if structData.NumField() == 2 {
				epoch := structData.Field(0).(*array.Int64).Int64Values()
				tzoffset := structData.Field(1).(*array.Int32).Int32Values()
				for i := range a.Len() {
					if a.IsNull(i) {
						tb.AppendNull()
						continue
					}
					l := snowflakeOffsetLocation(tzoffset[i])
					t := scaledTimestampToTime(epoch[i], scaleDivisor, fractionMultiplier).In(l)
					v, err := getArrowTimestampFromTime(t, dt.Unit, originalArrowUnit, maxTimestampPrecision)
					if err != nil {
						return nil, err
					}
					tb.Append(v)
				}
			} else {
				epoch := structData.Field(0).(*array.Int64).Int64Values()
				fraction := structData.Field(1).(*array.Int32).Int32Values()
				tzoffset := structData.Field(2).(*array.Int32).Int32Values()
				for i := range a.Len() {
					if a.IsNull(i) {
						tb.AppendNull()
						continue
					}
					l := snowflakeOffsetLocation(tzoffset[i])
					v, err := getArrowTimestampFromTime(time.Unix(epoch[i], int64(fraction[i])).In(l), dt.Unit, originalArrowUnit, maxTimestampPrecision)
					if err != nil {
						return nil, err
					}
					tb.Append(v)
				}
			}
			return tb.NewArray(), nil
		}
	}
}

// timestampEdgeValues returns int64 values around every boundary the
// conversion cares about at the given scale, plus ordinary dates.
func timestampEdgeValues(scale int64) []int64 {
	pow := int64(math.Pow10(int(scale)))
	vals := []int64{0, 1, -1, pow - 1, -pow + 1, pow, -pow, math.MaxInt64, math.MinInt64, math.MaxInt64 - 1, math.MinInt64 + 1}
	// Ordinary instants: 2024-06-01, 1969-12-31 23:59:59.x, 1600-01-01, 9999-12-31.
	for _, sec := range []int64{1717200000, -1, -11676096000, 253402300799} {
		vals = append(vals, sec*pow, sec*pow+pow/3, sec*pow-pow/7)
	}
	// Boundaries of the NanosecondsNoOverflow check and of each output unit.
	for _, sec := range []int64{9223372036, 9223372037, -9223372036, -9223372037, math.MaxInt64 / 1000, math.MinInt64 / 1000, math.MaxInt64 / 1_000_000, math.MinInt64 / 1_000_000} {
		for _, d := range []int64{-1, 0, 1} {
			s := sec + d
			if s > math.MaxInt64/pow || s < math.MinInt64/pow {
				continue
			}
			vals = append(vals, s*pow, s*pow+(pow-1))
			if s*pow >= math.MinInt64+pow {
				vals = append(vals, s*pow-1)
			}
		}
	}
	r := rand.New(rand.NewPCG(uint64(scale), 42))
	for range 50 {
		vals = append(vals, r.Int64(), -r.Int64(), r.Int64N(1<<40)*pow, -r.Int64N(1<<40)*pow+r.Int64N(pow))
	}
	return vals
}

var timestampFractions = []int32{0, 1, 999_999_999, 500_000_000, 123_456_789, -1, -999_999_999, 1_000_000_000, math.MaxInt32, math.MinInt32}

type timestampCase struct {
	name     string
	typ      string
	makeData func(mem memory.Allocator, scale int64) arrow.Array
}

func timestampCases() []timestampCase {
	int64Array := func(mem memory.Allocator, vals []int64) arrow.Array {
		b := array.NewInt64Builder(mem)
		defer b.Release()
		for i, v := range vals {
			if i%11 == 5 {
				b.AppendNull()
			}
			b.Append(v)
		}
		return b.NewArray()
	}
	structArray := func(mem memory.Allocator, fields []arrow.Field, appendRow func(sb *array.StructBuilder, i int), n int) arrow.Array {
		sb := array.NewStructBuilder(mem, arrow.StructOf(fields...))
		defer sb.Release()
		for i := range n {
			if i%11 == 5 {
				sb.AppendNull()
				for f := range sb.NumField() {
					sb.FieldBuilder(f).AppendNull()
				}
			}
			sb.Append(true)
			appendRow(sb, i)
		}
		return sb.NewArray()
	}
	epochFraction := func(withOffset bool) func(mem memory.Allocator, scale int64) arrow.Array {
		return func(mem memory.Allocator, scale int64) arrow.Array {
			epochs := timestampEdgeValues(0)
			fields := []arrow.Field{{Name: "epoch", Type: arrow.PrimitiveTypes.Int64}, {Name: "fraction", Type: arrow.PrimitiveTypes.Int32}}
			if withOffset {
				fields = append(fields, arrow.Field{Name: "timezone", Type: arrow.PrimitiveTypes.Int32})
			}
			n := len(epochs) * len(timestampFractions)
			return structArray(mem, fields, func(sb *array.StructBuilder, i int) {
				sb.FieldBuilder(0).(*array.Int64Builder).Append(epochs[i/len(timestampFractions)])
				sb.FieldBuilder(1).(*array.Int32Builder).Append(timestampFractions[i%len(timestampFractions)])
				if withOffset {
					sb.FieldBuilder(2).(*array.Int32Builder).Append(int32(1440 + (i%25-12)*60))
				}
			}, n)
		}
	}
	return []timestampCase{
		{"NTZ int64", "TIMESTAMP_NTZ", func(mem memory.Allocator, scale int64) arrow.Array {
			return int64Array(mem, timestampEdgeValues(scale))
		}},
		{"NTZ struct", "TIMESTAMP_NTZ", epochFraction(false)},
		{"LTZ int64", "TIMESTAMP_LTZ", func(mem memory.Allocator, scale int64) arrow.Array {
			return int64Array(mem, timestampEdgeValues(scale))
		}},
		{"LTZ struct", "TIMESTAMP_LTZ", epochFraction(false)},
		{"TZ struct2", "TIMESTAMP_TZ", func(mem memory.Allocator, scale int64) arrow.Array {
			vals := timestampEdgeValues(scale)
			fields := []arrow.Field{{Name: "epoch", Type: arrow.PrimitiveTypes.Int64}, {Name: "timezone", Type: arrow.PrimitiveTypes.Int32}}
			return structArray(mem, fields, func(sb *array.StructBuilder, i int) {
				sb.FieldBuilder(0).(*array.Int64Builder).Append(vals[i])
				sb.FieldBuilder(1).(*array.Int32Builder).Append(int32(1440 + (i%25-12)*60))
			}, len(vals))
		}},
		{"TZ struct3", "TIMESTAMP_TZ", epochFraction(true)},
	}
}

// TestTimestampTransformerMatchesLegacy checks the integer conversion against
// the previous time.Time-based implementation, value by value, for every
// representation, scale and precision mode. Each value is converted on its own
// so that a value the legacy code rejects doesn't hide the rest of the column.
func TestTimestampTransformerMatchesLegacy(t *testing.T) {
	loc, err := time.LoadLocation("America/New_York")
	require.NoError(t, err)
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	for _, tc := range timestampCases() {
		for scale := range int64(10) {
			for _, mode := range []MaxTimestampPrecision{Nanoseconds, NanosecondsNoOverflow, Microseconds} {
				t.Run(fmt.Sprintf("%s/scale=%d/%v", tc.name, scale, mode), func(t *testing.T) {
					var tz string
					var l *time.Location
					switch tc.typ {
					case "TIMESTAMP_LTZ":
						tz, l = loc.String(), loc
					case "TIMESTAMP_TZ":
						tz = "UTC"
					}
					dt := &arrow.TimestampType{Unit: getArrowTimeUnit(scale, mode), TimeZone: tz}
					orig := getArrowTimeUnit(scale, Nanoseconds)
					got := timestampTransformer(tc.typ == "TIMESTAMP_TZ", dt, scale, orig, mode, l)
					want := legacyTimestampTransformer(tc.typ, dt, scale, orig, mode, l)

					data := tc.makeData(mem, scale)
					defer data.Release()
					var okVals []int
					for i := range data.Len() {
						one := array.NewSlice(data, int64(i), int64(i+1))
						wantArr, wantErr := want(ctx, one)
						gotArr, gotErr := got(ctx, one)
						one.Release()
						if wantErr != nil {
							if assert.Error(t, gotErr, "row %d", i) {
								assert.Equal(t, wantErr.Error(), gotErr.Error(), "row %d", i)
							}
							if gotArr != nil {
								gotArr.Release()
							}
							continue
						}
						okVals = append(okVals, i)
						if assert.NoError(t, gotErr, "row %d", i) {
							assert.Truef(t, array.Equal(wantArr, gotArr), "row %d: want %v, got %v", i, wantArr, gotArr)
							assert.True(t, arrow.TypeEqual(wantArr.DataType(), gotArr.DataType()))
							gotArr.Release()
						}
						wantArr.Release()
					}
					require.NotEmpty(t, okVals)
				})
			}
		}
	}
}

// TestTimestampTransformerColumn converts whole columns, including nulls and a
// sliced (offset) input, and compares against the legacy implementation.
func TestTimestampTransformerColumn(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	r := rand.New(rand.NewPCG(1, 2))
	b := array.NewInt64Builder(mem)
	for i := range 1000 {
		if i%7 == 3 {
			b.AppendNull()
			continue
		}
		b.Append(1717200000_000000 + r.Int64N(1e15) - 5e14)
	}
	full := b.NewArray()
	b.Release()
	defer full.Release()

	for _, mode := range []MaxTimestampPrecision{Nanoseconds, NanosecondsNoOverflow, Microseconds} {
		for _, in := range []arrow.Array{full, array.NewSlice(full, 13, 997)} {
			dt := &arrow.TimestampType{Unit: getArrowTimeUnit(6, mode)}
			orig := getArrowTimeUnit(6, Nanoseconds)
			wantArr, err := legacyTimestampTransformer("TIMESTAMP_NTZ", dt, 6, orig, mode, nil)(ctx, in)
			require.NoError(t, err)
			gotArr, err := timestampTransformer(false, dt, 6, orig, mode, nil)(ctx, in)
			require.NoError(t, err)
			assert.Equal(t, in.NullN(), gotArr.NullN())
			assert.True(t, array.Equal(wantArr, gotArr))
			wantArr.Release()
			gotArr.Release()
			if in != full {
				in.Release()
			}
		}
	}
}

func BenchmarkTimestampTransformer(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	const n = 1 << 16
	ib := array.NewInt64Builder(memory.DefaultAllocator)
	for i := range n {
		if i%17 == 0 {
			ib.AppendNull()
			continue
		}
		ib.Append(1717200000_000000000 + r.Int64N(1e18) - 5e17)
	}
	compact := ib.NewArray()
	ib.Release()
	defer compact.Release()

	sb := array.NewStructBuilder(memory.DefaultAllocator, arrow.StructOf(
		arrow.Field{Name: "epoch", Type: arrow.PrimitiveTypes.Int64},
		arrow.Field{Name: "fraction", Type: arrow.PrimitiveTypes.Int32},
		arrow.Field{Name: "timezone", Type: arrow.PrimitiveTypes.Int32}))
	for range n {
		sb.Append(true)
		sb.FieldBuilder(0).(*array.Int64Builder).Append(1717200000 + r.Int64N(1e9) - 5e8)
		sb.FieldBuilder(1).(*array.Int32Builder).Append(r.Int32N(1e9))
		sb.FieldBuilder(2).(*array.Int32Builder).Append(1440 + 60)
	}
	tzStruct := sb.NewArray()
	sb.Release()
	defer tzStruct.Release()

	ctx := compute.WithAllocator(context.Background(), memory.DefaultAllocator)
	ntz := &arrow.TimestampType{Unit: arrow.Nanosecond}
	tz := &arrow.TimestampType{Unit: arrow.Nanosecond, TimeZone: "UTC"}
	cases := []struct {
		name string
		in   arrow.Array
		fn   colTransformer
	}{
		{"NTZ_scale9/Legacy", compact, legacyTimestampTransformer("TIMESTAMP_NTZ", ntz, 9, arrow.Nanosecond, Nanoseconds, nil)},
		{"NTZ_scale9/Integer", compact, timestampTransformer(false, ntz, 9, arrow.Nanosecond, Nanoseconds, nil)},
		{"LTZ_scale9/Legacy", compact, legacyTimestampTransformer("TIMESTAMP_LTZ", ntz, 9, arrow.Nanosecond, Nanoseconds, time.UTC)},
		{"LTZ_scale9/Integer", compact, timestampTransformer(false, ntz, 9, arrow.Nanosecond, Nanoseconds, time.UTC)},
		{"TZ_struct3/Legacy", tzStruct, legacyTimestampTransformer("TIMESTAMP_TZ", tz, 9, arrow.Nanosecond, Nanoseconds, nil)},
		{"TZ_struct3/Integer", tzStruct, timestampTransformer(true, tz, 9, arrow.Nanosecond, Nanoseconds, nil)},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				out, err := c.fn(ctx, c.in)
				if err != nil {
					b.Fatal(err)
				}
				out.Release()
			}
		})
	}
}
