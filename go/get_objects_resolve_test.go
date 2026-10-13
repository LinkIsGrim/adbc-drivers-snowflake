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
	"testing"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/snowflakedb/gosnowflake/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveGetObjectsScope(t *testing.T) {
	sp := func(s string) *string { return &s }
	tests := []struct {
		name              string
		depth             adbc.ObjectDepth
		catalog, dbSchema *string
		wantCatalog       *string
		wantSchema        *string
		wantEmpty         bool
		wantQueries       int
		matchExactly      bool
	}{
		{"plain names need no resolution", adbc.ObjectDepthAll, sp("DB"), sp("SCH"), sp("DB"), sp("SCH"), false, 0, false},
		{"catalogs depth never resolves", adbc.ObjectDepthCatalogs, sp("MY_DB"), nil, nil, nil, false, 0, false},
		{"underscore catalog and schema resolve", adbc.ObjectDepthAll, sp("MY_DB"), sp("MY_SCH"), sp("MY_DB"), sp("MY_SCH"), false, 2, false},
		{"schema is not resolved at db_schemas depth", adbc.ObjectDepthDBSchemas, sp("MY_DB"), sp("MY_SCH"), sp("MY_DB"), nil, false, 1, false},
		{"percent patterns are not resolved", adbc.ObjectDepthAll, sp("MY_%"), sp("MY_SCH"), nil, nil, false, 0, false},
		{"several matching databases fall back", adbc.ObjectDepthAll, sp("MULTI_DB"), sp("MY_SCH"), nil, nil, false, 1, false},
		{"no matching database is an empty result", adbc.ObjectDepthTables, sp("NO_DB"), nil, nil, nil, true, 1, false},
		{"no matching schema falls back", adbc.ObjectDepthTables, sp("DB"), sp("NO_SCH"), sp("DB"), nil, false, 1, false},
		{"match exactly takes names as is", adbc.ObjectDepthAll, sp("MY_DB"), sp("MY_SCH"), sp("MY_DB"), sp("MY_SCH"), false, 0, true},
		{"match exactly empty schema scopes to the database", adbc.ObjectDepthTables, sp("MY_DB"), sp(""), sp("MY_DB"), nil, false, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fc := &fakeConn{Respond: fakeAccountShowTerse}
			scope, empty, err := resolveGetObjectsScope(context.Background(), fc, tt.depth, tt.catalog, tt.dbSchema, tt.matchExactly)
			require.NoError(t, err)
			assert.Equal(t, tt.wantEmpty, empty)
			assert.Equal(t, tt.wantCatalog, scope.catalog)
			assert.Equal(t, tt.wantSchema, scope.dbSchema)
			assert.Len(t, fc.Statements(), tt.wantQueries)
		})
	}
}

func TestResolveGetObjectsScopeShowNoMatch(t *testing.T) {
	sp := func(s string) *string { return &s }
	fc := &fakeConn{Respond: func(string) (driver.Rows, error) {
		return nil, &gosnowflake.SnowflakeError{Number: errShowNoMatch}
	}}
	scope, empty, err := resolveGetObjectsScope(context.Background(), fc, adbc.ObjectDepthAll, sp("MY_DB"), nil, false)
	require.NoError(t, err)
	assert.True(t, empty)
	assert.Empty(t, scope.dbQueryID)
}

func TestResolveGetObjectsScopeError(t *testing.T) {
	sp := func(s string) *string { return &s }
	fc := &fakeConn{Respond: func(string) (driver.Rows, error) {
		return nil, &gosnowflake.SnowflakeError{Number: 390114, Message: "auth token expired"}
	}}
	_, _, err := resolveGetObjectsScope(context.Background(), fc, adbc.ObjectDepthAll, sp("MY_DB"), nil, false)
	assert.ErrorContains(t, err, "auth token expired")
}
