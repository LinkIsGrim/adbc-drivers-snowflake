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
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// copyResultConn answers COPY queries. If results is non-empty, each COPY
// returns the next canned result. Otherwise it behaves like Snowflake: each
// COPY loads every staged file it has not loaded before. Only QueryContext is
// implemented.
type copyResultConn struct {
	snowflakeConn

	mu      sync.Mutex
	results []*copyResultRows
	staged  map[string]string // file -> row count
	loaded  map[string]bool
	queries int
}

// stage makes a file visible to later COPYs, like an upload to the stage.
func (c *copyResultConn) stage(file, rows string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.staged == nil {
		c.staged, c.loaded = map[string]string{}, map[string]bool{}
	}
	c.staged[file] = rows
}

func (c *copyResultConn) QueryContext(_ context.Context, _ string, _ []driver.NamedValue) (driver.Rows, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queries++
	if len(c.results) > 0 {
		r := c.results[0]
		c.results = c.results[1:]
		return r, nil
	}
	pending := map[string]string{}
	for f, n := range c.staged {
		if !c.loaded[f] {
			pending[f] = n
			c.loaded[f] = true
		}
	}
	if len(pending) == 0 {
		return &copyResultRows{cols: []string{"status"}, rows: [][]driver.Value{{"Copy executed with 0 files processed."}}}, nil
	}
	return copyResult(pending), nil
}

type copyResultRows struct {
	cols []string
	rows [][]driver.Value
	err  error
}

func (r *copyResultRows) Columns() []string { return r.cols }
func (r *copyResultRows) Close() error      { return nil }
func (r *copyResultRows) Next(dest []driver.Value) error {
	if len(r.rows) == 0 {
		if r.err != nil {
			return r.err
		}
		return io.EOF
	}
	copy(dest, r.rows[0])
	r.rows = r.rows[1:]
	return nil
}

var copyResultColumns = []string{"file", "status", "rows_parsed", "rows_loaded", "error_limit", "errors_seen"}

func copyResult(files map[string]string) *copyResultRows {
	r := &copyResultRows{cols: copyResultColumns}
	for f, n := range files {
		r.rows = append(r.rows, []driver.Value{"adbc$bind/" + f, "LOADED", n, n, "1", "0"})
	}
	return r
}

func TestExecuteCopyQueryReturnsRowsLoaded(t *testing.T) {
	cn := &copyResultConn{results: []*copyResultRows{copyResult(map[string]string{"0.parquet": "100", "1.parquet": "23"})}}
	var files fileSet
	files.Add("0.parquet")
	files.Add("1.parquet")
	files.Add("2.parquet")

	n, err := executeCopyQuery(context.Background(), cn, copyQuery, "T", &files)
	require.NoError(t, err)
	assert.EqualValues(t, 123, n)
	assert.Equal(t, 1, files.Len(), "loaded files should be removed")
}

func TestExecuteCopyQueryNoFiles(t *testing.T) {
	n, err := executeCopyQuery(context.Background(), &copyResultConn{}, copyQuery, "T", &fileSet{})
	require.NoError(t, err)
	assert.Zero(t, n)
}

func TestExecuteCopyQueryErrors(t *testing.T) {
	t.Run("missing rows_loaded", func(t *testing.T) {
		cn := &copyResultConn{results: []*copyResultRows{{cols: []string{"file", "status"}, rows: [][]driver.Value{{"0.parquet", "LOADED"}}}}}
		_, err := executeCopyQuery(context.Background(), cn, copyQuery, "T", &fileSet{})
		assert.ErrorContains(t, err, "rows_loaded")
	})
	t.Run("unparsable rows_loaded", func(t *testing.T) {
		cn := &copyResultConn{results: []*copyResultRows{copyResult(map[string]string{"0.parquet": "lots"})}}
		_, err := executeCopyQuery(context.Background(), cn, copyQuery, "T", &fileSet{})
		assert.ErrorContains(t, err, "rows_loaded")
	})
	t.Run("rows counted before a failure are returned", func(t *testing.T) {
		r := copyResult(map[string]string{"0.parquet": "7"})
		r.err = errors.New("network")
		cn := &copyResultConn{results: []*copyResultRows{r}}
		n, err := executeCopyQuery(context.Background(), cn, copyQuery, "T", &fileSet{})
		assert.Error(t, err)
		assert.EqualValues(t, 7, n)
	})
}

func TestRunCopyTasksAccumulatesRowsLoaded(t *testing.T) {
	cn := &copyResultConn{}
	var rowsLoaded atomic.Int64
	ready, finish, _ := runCopyTasks(context.Background(), cn, copyQuery, "T", 2, &rowsLoaded)
	for i, f := range []string{"0.parquet", "1.parquet", "2.parquet", "3.parquet"} {
		cn.stage(f, []string{"10", "20", "30", "40"}[i])
		ready(f)
	}
	require.NoError(t, finish())
	// However the background and final COPYs interleave, every file is
	// loaded exactly once.
	assert.EqualValues(t, 100, rowsLoaded.Load())
}
