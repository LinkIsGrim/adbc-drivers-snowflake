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
	"io"
	"sync"
	"testing"

	"github.com/adbc-drivers/driverbase-go/driverbase"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/snowflakedb/gosnowflake/v2"
	"github.com/stretchr/testify/require"
)

// recordedStatement is one statement sent through a fakeConn.
type recordedStatement struct {
	Query string
	Args  []driver.NamedValue
}

// fakeConn is a snowflakeConn that records every statement it receives and
// answers each with an empty result set, or with whatever Respond returns.
// It lets tests observe the SQL a code path issues without a Snowflake account.
type fakeConn struct {
	// Respond, if set, overrides the default empty result for a query.
	// Returning nil rows and a nil error falls back to the default.
	Respond func(query string) (driver.Rows, error)

	mu         sync.Mutex
	statements []recordedStatement
}

func (c *fakeConn) record(q string, args []driver.NamedValue) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.statements = append(c.statements, recordedStatement{Query: q, Args: args})
}

// Statements returns a copy of the statements received so far, in arrival order.
func (c *fakeConn) Statements() []recordedStatement {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]recordedStatement(nil), c.statements...)
}

func (c *fakeConn) QueryContext(_ context.Context, q string, args []driver.NamedValue) (driver.Rows, error) {
	c.record(q, args)
	if c.Respond != nil {
		if rows, err := c.Respond(q); rows != nil || err != nil {
			return rows, err
		}
	}
	return &fakeRows{}, nil
}

func (c *fakeConn) ExecContext(_ context.Context, q string, args []driver.NamedValue) (driver.Result, error) {
	c.record(q, args)
	return driver.RowsAffected(0), nil
}

func (c *fakeConn) Ping(context.Context) error { return nil }
func (c *fakeConn) Close() error               { return nil }

func (c *fakeConn) Prepare(string) (driver.Stmt, error) {
	panic("fakeConn: Prepare not implemented")
}

func (c *fakeConn) PrepareContext(context.Context, string) (driver.Stmt, error) {
	panic("fakeConn: PrepareContext not implemented")
}

func (c *fakeConn) Begin() (driver.Tx, error) {
	panic("fakeConn: Begin not implemented")
}

func (c *fakeConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	panic("fakeConn: BeginTx not implemented")
}

func (c *fakeConn) QueryArrowStream(context.Context, string, ...driver.NamedValue) (gosnowflake.ArrowStreamLoader, error) {
	panic("fakeConn: QueryArrowStream not implemented")
}

// fakeRows is a gosnowflake.SnowflakeRows that yields Values in order.
type fakeRows struct {
	Cols   []string
	Values [][]driver.Value
	// Err, if set, is returned by Next once Values are exhausted instead of io.EOF.
	Err error

	next int
}

func (r *fakeRows) Columns() []string {
	if r.Cols == nil {
		return []string{"result"}
	}
	return r.Cols
}

func (r *fakeRows) Close() error { return nil }

func (r *fakeRows) Next(dest []driver.Value) error {
	if r.next >= len(r.Values) {
		if r.Err != nil {
			return r.Err
		}
		return io.EOF
	}
	copy(dest, r.Values[r.next])
	r.next++
	return nil
}

func (r *fakeRows) GetQueryID() string                 { return "fake-query-id" }
func (r *fakeRows) GetStatus() gosnowflake.QueryStatus { return gosnowflake.QueryStatusComplete }
func (r *fakeRows) NextResultSet() error               { return io.EOF }

// newFakeConnection returns a connectionImpl backed by a fakeConn.
func newFakeConnection(t *testing.T) (*connectionImpl, *fakeConn) {
	t.Helper()
	fc := &fakeConn{}
	drv := driverbase.NewDriverImplBase(driverbase.DefaultDriverInfo("Snowflake"), memory.DefaultAllocator)
	db, err := driverbase.NewDatabaseImplBase(context.Background(), &drv, driverbase.TracingOptions{})
	require.NoError(t, err)
	return &connectionImpl{cn: fc, ConnectionImplBase: driverbase.NewConnectionImplBase(&db)}, fc
}
