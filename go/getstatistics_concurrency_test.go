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
	"database/sql/driver"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/adbc-drivers/driverbase-go/driverbase"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// statsConn fakes the metadata queries GetStatistics issues against an
// account with databases DB0..DB4, each with tables T0..T2 in schemas S0 and
// S1, where T1 is clustered. Every query takes a few milliseconds, and the
// connection records how many were in flight at once. Only QueryContext is
// implemented.
type statsConn struct {
	snowflakeConn
	latency time.Duration
	// tablesPerSchema defaults to 3.
	tablesPerSchema int

	inFlight, maxInFlight atomic.Int32
	mu                    sync.Mutex
	queries               []string
}

type statsRows struct {
	cols []string
	rows [][]driver.Value
}

func (r *statsRows) Columns() []string { return r.cols }
func (r *statsRows) Close() error      { return nil }
func (r *statsRows) Next(dest []driver.Value) error {
	if len(r.rows) == 0 {
		return io.EOF
	}
	copy(dest, r.rows[0])
	r.rows = r.rows[1:]
	return nil
}

func (c *statsConn) QueryContext(ctx context.Context, q string, _ []driver.NamedValue) (driver.Rows, error) {
	n := c.inFlight.Add(1)
	defer c.inFlight.Add(-1)
	for {
		m := c.maxInFlight.Load()
		if n <= m || c.maxInFlight.CompareAndSwap(m, n) {
			break
		}
	}
	c.mu.Lock()
	c.queries = append(c.queries, q)
	c.mu.Unlock()
	select {
	case <-time.After(c.latency):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	switch {
	case strings.HasPrefix(q, "SHOW DATABASES"):
		r := &statsRows{cols: []string{"created_on", "name"}}
		for d := range 5 {
			r.rows = append(r.rows, []driver.Value{"2026-01-01", fmt.Sprintf("DB%d", d)})
		}
		return r, nil
	case strings.Contains(q, "information_schema.tables"):
		var db string
		_, _ = fmt.Sscanf(q[strings.Index(q, `FROM "`)+len(`FROM "`):], "DB%s", &db)
		db = "DB" + db[:1]
		r := &statsRows{cols: []string{"table_catalog", "table_schema", "table_name", "row_count", "bytes", "retention_time", "clustering_key"}}
		n := c.tablesPerSchema
		if n == 0 {
			n = 3
		}
		for s := range 2 {
			for tb := range n {
				var key driver.Value
				if tb == 1 {
					key = "LINEAR(C)"
				}
				r.rows = append(r.rows, []driver.Value{db, fmt.Sprintf("S%d", s), fmt.Sprintf("T%d", tb), fmt.Sprint(100 + tb), fmt.Sprint(1000 + tb), "1", key})
			}
		}
		return r, nil
	case strings.Contains(q, "table_storage_metrics"):
		r := &statsRows{cols: []string{"table_schema", "table_name", "active_bytes", "time_travel_bytes", "failsafe_bytes"}}
		for s := range 2 {
			for tb := range 3 {
				r.rows = append(r.rows, []driver.Value{fmt.Sprintf("S%d", s), fmt.Sprintf("T%d", tb), "10", "20", "30"})
			}
		}
		return r, nil
	case strings.Contains(q, "SYSTEM$CLUSTERING_INFORMATION"):
		return &statsRows{cols: []string{"info"}, rows: [][]driver.Value{{`{"average_depth": 2.5}`}}}, nil
	}
	return nil, fmt.Errorf("statsConn: unexpected query %q", q)
}

// dumpStatistics renders a GetStatistics result as one line per statistic.
func dumpStatistics(t *testing.T, rdr array.RecordReader) string {
	var sb strings.Builder
	for rdr.Next() {
		rec := rdr.RecordBatch()
		js, err := rec.MarshalJSON()
		require.NoError(t, err)
		sb.Write(js)
	}
	require.NoError(t, rdr.Err())
	return sb.String()
}

func newStatsConnection(t *testing.T, cn snowflakeConn) *connectionImpl {
	t.Helper()
	drv := driverbase.NewDriverImplBase(driverbase.DefaultDriverInfo("Snowflake"), memory.DefaultAllocator)
	db, err := driverbase.NewDatabaseImplBase(context.Background(), &drv, driverbase.TracingOptions{})
	require.NoError(t, err)
	return &connectionImpl{cn: cn, ConnectionImplBase: driverbase.NewConnectionImplBase(&db)}
}

func TestGetStatisticsRunsQueriesConcurrently(t *testing.T) {
	for _, approximate := range []bool{true, false} {
		t.Run(fmt.Sprintf("approximate=%v", approximate), func(t *testing.T) {
			cn := &statsConn{latency: 20 * time.Millisecond}
			rdr, err := newStatsConnection(t, cn).GetStatistics(context.Background(), nil, nil, nil, approximate)
			require.NoError(t, err)
			defer rdr.Release()
			got := dumpStatistics(t, rdr)

			golden := fmt.Sprintf("testdata/get_statistics/approximate_%v.json", approximate)
			if os.Getenv("UPDATE_GET_STATISTICS_GOLDEN") != "" {
				require.NoError(t, os.WriteFile(golden, []byte(got+"\n"), 0o644))
			}
			want, err := os.ReadFile(golden)
			require.NoError(t, err)
			assert.JSONEq(t, string(want), got, "GetStatistics output changed")

			// SHOW DATABASES, one tables query per database, and in exact
			// mode one storage-metrics query per database plus one
			// clustering query per clustered table.
			wantQueries := 1 + 5
			if !approximate {
				wantQueries += 5 + 5*2
			}
			assert.Len(t, cn.queries, wantQueries)
			assert.Greater(t, cn.maxInFlight.Load(), int32(1), "queries did not overlap")
			assert.LessOrEqual(t, cn.maxInFlight.Load(), int32(getStatisticsConcurrency))
		})
	}
}

func TestGetStatisticsExactLimit(t *testing.T) {
	// 5 databases x 2 schemas x 101 tables crosses the limit only once most
	// of the per-database queries have finished, in whatever order.
	cn := &statsConn{latency: time.Millisecond, tablesPerSchema: 101}
	_, err := newStatsConnection(t, cn).GetStatistics(context.Background(), nil, nil, nil, false)
	assert.ErrorContains(t, err, fmt.Sprintf("exact statistics requested for more than %d tables", maxExactTables))

	// Approximate mode has no limit.
	rdr, err := newStatsConnection(t, cn).GetStatistics(context.Background(), nil, nil, nil, true)
	require.NoError(t, err)
	rdr.Release()
}
