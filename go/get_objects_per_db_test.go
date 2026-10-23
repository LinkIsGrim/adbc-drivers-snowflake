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
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/adbc-drivers/driverbase-go/driverbase"
	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/snowflakedb/gosnowflake/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestForEachCatalog(t *testing.T) {
	catalogs := make([]string, 30)
	for i := range catalogs {
		catalogs[i] = fmt.Sprintf("DB%02d", i)
	}
	var running, peak atomic.Int32
	infos, err := forEachCatalog(context.Background(), catalogs, func(_ context.Context, catalog string) (driverbase.GetObjectsInfo, error) {
		n := running.Add(1)
		defer running.Add(-1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		return driverbase.GetObjectsInfo{CatalogName: &catalog}, nil
	})
	require.NoError(t, err)
	require.Len(t, infos, len(catalogs))
	for i, info := range infos {
		assert.Equal(t, catalogs[i], *info.CatalogName, "results keep catalog order")
	}
	assert.LessOrEqual(t, peak.Load(), int32(getObjectsConcurrency))
	assert.Greater(t, peak.Load(), int32(1), "catalogs are queried concurrently")

	boom := errors.New("boom")
	_, err = forEachCatalog(context.Background(), catalogs, func(_ context.Context, catalog string) (driverbase.GetObjectsInfo, error) {
		if catalog == "DB07" {
			return driverbase.GetObjectsInfo{}, boom
		}
		return driverbase.GetObjectsInfo{CatalogName: &catalog}, nil
	})
	assert.ErrorIs(t, err, boom)
}

// fakeShowAccount answers SHOW TERSE DATABASES / SCHEMAS / OBJECTS like an
// account holding the given databases, each with schemas PUBLIC and OTHER,
// each with tables T1 and T2. LIKE patterns are matched case-insensitively
// with '_' as a single-character wildcard.
func fakeShowAccount(databases ...string) func(string) (driver.Rows, error) {
	like := func(pattern, name string) bool {
		if pattern == "" {
			return true
		}
		// An escaped underscore matches only itself.
		pattern = strings.ReplaceAll(pattern, `\_`, "\x00")
		if len(pattern) != len(name) {
			return false
		}
		for i := range pattern {
			switch {
			case pattern[i] == '_':
			case pattern[i] == 0:
				if name[i] != '_' {
					return false
				}
			case !strings.EqualFold(pattern[i:i+1], name[i:i+1]):
				return false
			}
		}
		return true
	}
	cols := []string{"created_on", "name", "kind", "database_name", "schema_name"}
	return func(query string) (driver.Rows, error) {
		m := reShowTerse.FindStringSubmatch(query)
		if m == nil {
			if !strings.HasPrefix(query, "SHOW TERSE /* ADBC:getObjects */ OBJECTS") {
				return nil, nil
			}
			rows := &fakeRows{Cols: cols}
			for _, db := range databases {
				for _, sch := range []string{"PUBLIC", "OTHER"} {
					if strings.Contains(query, "IN DATABASE "+quoteIdentifier(db)) || strings.Contains(query, "IN SCHEMA "+quoteIdentifier(db)+"."+quoteIdentifier(sch)) {
						for _, tbl := range []string{"T1", "T2"} {
							rows.Values = append(rows.Values, []driver.Value{"", tbl, "TABLE", db, sch})
						}
					}
				}
			}
			return rows, nil
		}
		rows := &fakeRows{Cols: cols}
		switch m[1] {
		case objDatabases:
			for _, db := range databases {
				if like(m[2], db) {
					rows.Values = append(rows.Values, []driver.Value{"", db, "", "", ""})
				}
			}
		case objSchemas:
			for _, db := range databases {
				if strings.HasSuffix(query, "IN DATABASE "+quoteIdentifier(db)) {
					for _, sch := range []string{"PUBLIC", "OTHER"} {
						if like(m[2], sch) {
							rows.Values = append(rows.Values, []driver.Value{"", sch, "", db, ""})
						}
					}
				}
			}
		}
		return rows, nil
	}
}

func readGetObjects(t *testing.T, rdr array.RecordReader) string {
	t.Helper()
	defer rdr.Release()
	var out strings.Builder
	for rdr.Next() {
		b, err := array.RecordToStructArray(rdr.RecordBatch()).MarshalJSON()
		require.NoError(t, err)
		out.Write(b)
	}
	require.NoError(t, rdr.Err())
	return out.String()
}

// Tables depth merges the per-database results in database order, and a
// catalog pattern matching several databases no longer scans the account.
func TestGetObjectsTablesAcrossDatabases(t *testing.T) {
	sp := func(s string) *string { return &s }
	cnxn, fc := newFakeConnection(t)
	fc.Respond = fakeShowAccount("MY_DB", "MYXDB", "OTHER_DB")

	rdr, err := cnxn.GetObjects(context.Background(), adbc.ObjectDepthTables, sp("MY_DB"), sp("public"), nil, nil, nil)
	require.NoError(t, err)
	got := readGetObjects(t, rdr)

	assert.JSONEq(t, `[{"catalog_name":"MY_DB","catalog_db_schemas":[{"db_schema_name":"PUBLIC","db_schema_tables":[{"table_name":"T1","table_type":"TABLE","table_columns":null,"table_constraints":null},{"table_name":"T2","table_type":"TABLE","table_columns":null,"table_constraints":null}]}]}`+
		`,{"catalog_name":"MYXDB","catalog_db_schemas":[{"db_schema_name":"PUBLIC","db_schema_tables":[{"table_name":"T1","table_type":"TABLE","table_columns":null,"table_constraints":null},{"table_name":"T2","table_type":"TABLE","table_columns":null,"table_constraints":null}]}]}]`, got)
	for _, st := range fc.Statements() {
		assert.NotContains(t, st.Query, "IN ACCOUNT LIKE", st.Query)
		if !strings.Contains(st.Query, objDatabases) {
			assert.NotContains(t, st.Query, "IN ACCOUNT", st.Query)
		}
	}
	// The schema pattern matched only PUBLIC, so tables are listed in that
	// schema with its real (upper-case) name.
	assert.Contains(t, queries(fc), `SHOW TERSE /* ADBC:getObjects */ OBJECTS IN SCHEMA "MY_DB"."PUBLIC"`)
}

func queries(fc *fakeConn) []string {
	var qs []string
	for _, st := range fc.Statements() {
		qs = append(qs, st.Query)
	}
	return qs
}

// At Columns depth every database is read through its own information_schema.
func TestGetObjectsColumnsPerDatabase(t *testing.T) {
	sp := func(s string) *string { return &s }
	cnxn, fc := newFakeConnection(t)
	fc.Respond = fakeShowAccount("MY_DB", "MYXDB")

	rdr, err := cnxn.GetObjects(context.Background(), adbc.ObjectDepthAll, sp("MY_DB"), sp("PUBLIC"), sp("T1"), nil, nil)
	require.NoError(t, err)
	got := readGetObjects(t, rdr)
	assert.JSONEq(t, `[{"catalog_name":"MY_DB","catalog_db_schemas":[]},{"catalog_name":"MYXDB","catalog_db_schemas":[]}]`, got)

	qs := queries(fc)
	for _, db := range []string{"MY_DB", "MYXDB"} {
		q := quoteIdentifier(db)
		assert.Contains(t, qs, fmt.Sprintf(queryGetObjectsAll, q))
		assert.Contains(t, qs, `SHOW PRIMARY KEYS /* ADBC:getObjectsTables */ IN SCHEMA `+q+`."PUBLIC"`)
		assert.Contains(t, qs, `SHOW COLUMNS /* ADBC:getObjects */ IN TABLE `+q+`."PUBLIC"."T1"`)
	}
	for _, st := range fc.Statements() {
		if strings.Contains(st.Query, "information_schema.columns") {
			var dbName any
			for _, a := range st.Args {
				if a.Name == "DB_NAME" {
					dbName = a.Value
				}
			}
			assert.Contains(t, st.Query, fmt.Sprintf("FROM %s.information_schema.columns", quoteIdentifier(dbName.(string))))
		}
		if !strings.Contains(st.Query, objDatabases) {
			assert.NotContains(t, st.Query, "IN ACCOUNT", st.Query)
		}
	}
}

// Above getObjectsMaxScopedCatalogs databases, Columns depth shares one set of
// account-wide SHOWs but still reads each database's information_schema.
func TestGetObjectsColumnsManyDatabases(t *testing.T) {
	var dbs []string
	for i := range getObjectsMaxScopedCatalogs + 1 {
		dbs = append(dbs, fmt.Sprintf("DB%02d", i))
	}
	cnxn, fc := newFakeConnection(t)
	fc.Respond = fakeShowAccount(dbs...)

	rdr, err := cnxn.GetObjects(context.Background(), adbc.ObjectDepthAll, nil, nil, nil, nil, nil)
	require.NoError(t, err)
	readGetObjects(t, rdr)

	var inAccount, perDB int
	for _, q := range queries(fc) {
		switch {
		case strings.HasPrefix(q, "SHOW") && strings.HasSuffix(q, " IN ACCOUNT"):
			inAccount++
		case strings.Contains(q, "information_schema.columns"):
			perDB++
		}
	}
	assert.Equal(t, 5, inAccount, "SHOW DATABASES plus one shared SHOW each for primary, imported and unique keys and columns")
	assert.Equal(t, len(dbs), perDB)
}

// A database whose information_schema can't be read comes back without
// schemas instead of failing the whole call.
func TestGetObjectsColumnsUnreadableDatabase(t *testing.T) {
	sp := func(s string) *string { return &s }
	cnxn, fc := newFakeConnection(t)
	show := fakeShowAccount("SHARED_DB")
	fc.Respond = func(q string) (driver.Rows, error) {
		if strings.Contains(q, "information_schema.columns") {
			return nil, &gosnowflake.SnowflakeError{Number: errObjectNotFound}
		}
		return show(q)
	}
	rdr, err := cnxn.GetObjects(context.Background(), adbc.ObjectDepthAll, sp("SHARED_DB"), nil, nil, nil, nil)
	require.NoError(t, err)
	assert.JSONEq(t, `[{"catalog_name":"SHARED_DB","catalog_db_schemas":[]}]`, readGetObjects(t, rdr))
}

func TestGetObjectsNoMatchingDatabase(t *testing.T) {
	sp := func(s string) *string { return &s }
	for _, depth := range []adbc.ObjectDepth{adbc.ObjectDepthDBSchemas, adbc.ObjectDepthTables, adbc.ObjectDepthAll} {
		cnxn, fc := newFakeConnection(t)
		fc.Respond = func(string) (driver.Rows, error) {
			return nil, &gosnowflake.SnowflakeError{Number: errShowNoMatch}
		}
		rdr, err := cnxn.GetObjects(context.Background(), depth, sp("NO_DB"), nil, nil, nil, nil)
		require.NoError(t, err)
		assert.JSONEq(t, "[]", readGetObjects(t, rdr))
		assert.Len(t, fc.Statements(), 1, "only the SHOW DATABASES runs")
	}
}

func TestGetObjectsResolveError(t *testing.T) {
	sp := func(s string) *string { return &s }
	cnxn, fc := newFakeConnection(t)
	fc.Respond = func(string) (driver.Rows, error) {
		return nil, &gosnowflake.SnowflakeError{Number: 390114, Message: "auth token expired"}
	}
	_, err := cnxn.GetObjects(context.Background(), adbc.ObjectDepthTables, sp("MY_DB"), nil, nil, nil, nil)
	assert.ErrorContains(t, err, "auth token expired")
}
