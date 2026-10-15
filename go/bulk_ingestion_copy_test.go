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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stagedCopyConn answers COPY queries like Snowflake: each COPY loads the
// visible staged files it has not loaded before. Only QueryContext is
// implemented.
type stagedCopyConn struct {
	snowflakeConn

	mu      sync.Mutex
	visible []string
	loaded  map[string]bool
	copies  int
}

func (c *stagedCopyConn) show(file string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.visible = append(c.visible, file)
}

func (c *stagedCopyConn) copyCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.copies
}

func (c *stagedCopyConn) QueryContext(ctx context.Context, _ string, _ []driver.NamedValue) (driver.Rows, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.copies++
	if c.loaded == nil {
		c.loaded = map[string]bool{}
	}
	rows := &copyRows{}
	for _, f := range c.visible {
		if !c.loaded[f] {
			c.loaded[f] = true
			rows.files = append(rows.files, "stage/"+f)
		}
	}
	return rows, nil
}

// copyRows is a COPY result: one row per loaded file, or the single status
// column Snowflake returns when no files were loaded.
type copyRows struct {
	files []string
	i     int
}

func (r *copyRows) Columns() []string {
	if len(r.files) == 0 {
		return []string{"status"}
	}
	return []string{"file", "status"}
}

func (r *copyRows) Close() error { return nil }

func (r *copyRows) Next(dest []driver.Value) error {
	if len(r.files) == 0 {
		if r.i > 0 {
			return io.EOF
		}
		r.i++
		dest[0] = "Copy executed with 0 files processed."
		return nil
	}
	if r.i >= len(r.files) {
		return io.EOF
	}
	dest[0], dest[1] = r.files[r.i], "LOADED"
	r.i++
	return nil
}

// A file that no background COPY picked up is loaded by the final COPY that
// finishCopy runs. That COPY used to run on the errgroup's context, which is
// already canceled once Wait returns, so ingestion failed with "context
// canceled" and left the remaining files unloaded.
func TestFinishCopyLoadsRemainingFiles(t *testing.T) {
	cn := &stagedCopyConn{}
	fileReady, finishCopy, _ := runCopyTasks(context.Background(), cn, copyQuery, "T", 1)

	// The background COPY triggered by this file runs before the file is
	// visible on the stage, so it loads nothing.
	fileReady("0.parquet")
	require.Eventually(t, func() bool { return cn.copyCount() == 1 }, 5*time.Second, time.Millisecond)
	cn.show("0.parquet")

	require.NoError(t, finishCopy())
	assert.Equal(t, 2, cn.copyCount())
	assert.True(t, cn.loaded["0.parquet"])
}

func TestFinishCopyNothingRemaining(t *testing.T) {
	cn := &stagedCopyConn{}
	cn.show("0.parquet")
	fileReady, finishCopy, _ := runCopyTasks(context.Background(), cn, copyQuery, "T", 1)

	fileReady("0.parquet")
	require.Eventually(t, func() bool { return cn.copyCount() == 1 }, 5*time.Second, time.Millisecond)

	require.NoError(t, finishCopy())
	assert.Equal(t, 1, cn.copyCount(), "no final COPY is needed when every file was loaded")
}

func TestCancelCopyStopsFinalCopy(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cn := &stagedCopyConn{}
	fileReady, finishCopy, _ := runCopyTasks(ctx, cn, copyQuery, "T", 1)

	fileReady("0.parquet")
	require.Eventually(t, func() bool { return cn.copyCount() == 1 }, 5*time.Second, time.Millisecond)
	cn.show("0.parquet")

	// Canceling the caller's context still stops the final COPY.
	cancel()
	assert.ErrorIs(t, finishCopy(), context.Canceled)
	assert.Equal(t, 1, cn.copyCount())
}
