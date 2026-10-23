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
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var reGetObjectsAll = regexp.MustCompile(`FROM ("(?:[^"]|"")*")\.information_schema\.columns`)

var updateGolden = flag.Bool("update", false, "rewrite golden files under testdata/ instead of comparing against them")

// describeStatement renders a statement for a query-plan snapshot. SHOW and
// other short statements are shown verbatim; the embedded GetObjects templates
// are shown by file name along with their bound arguments.
func describeStatement(st recordedStatement) string {
	templates := []struct{ name, sql string }{
		{"queries/get_objects_dbschemas.sql", queryGetObjectsDbSchemas},
		{"queries/get_objects_tables.sql", queryGetObjectsTables},
		{"queries/get_objects_terse_catalogs.sql", queryGetObjectsTerseCatalogs},
	}
	desc := st.Query
	if m := reGetObjectsAll.FindStringSubmatch(st.Query); m != nil {
		all := fmt.Sprintf(queryGetObjectsAll, m[1])
		switch st.Query {
		case all:
			desc = "<queries/get_objects_all.sql in " + m[1] + ">"
		case strings.ReplaceAll(all, " ILIKE ", " LIKE "):
			desc = "<queries/get_objects_all.sql in " + m[1] + " with LIKE>"
		}
	}
	for _, tmpl := range templates {
		if st.Query == tmpl.sql {
			desc = "<" + tmpl.name + ">"
			break
		}
		// match_exactly runs the templates with LIKE in place of ILIKE.
		if st.Query == strings.ReplaceAll(tmpl.sql, " ILIKE ", " LIKE ") {
			desc = "<" + tmpl.name + " with LIKE>"
			break
		}
	}
	if len(st.Args) == 0 {
		return desc
	}
	args := make([]string, len(st.Args))
	for i, a := range st.Args {
		switch v := a.Value.(type) {
		case nil:
			args[i] = a.Name + "=NULL"
		case string:
			if strings.HasSuffix(a.Name, "_QUERY_ID") && v != "" {
				v = "<id>"
			}
			args[i] = fmt.Sprintf("%s=%q", a.Name, v)
		default:
			args[i] = fmt.Sprintf("%s=%v", a.Name, v)
		}
	}
	return desc + "\n        " + strings.Join(args, " ")
}

// TestGetObjectsQueryPlan snapshots the statements GetObjects sends for each
// depth across a set of filter shapes. It asserts on the SQL, not on results,
// so it runs without a Snowflake account. Several statements are issued
// concurrently, so each plan is sorted before comparing.
//
// Regenerate with: go test -run TestGetObjectsQueryPlan -update
func TestGetObjectsQueryPlan(t *testing.T) {
	sp := func(s string) *string { return &s }
	inputs := []struct {
		name                         string
		catalog, dbSchema, tableName *string
		columnName                   *string
		tableType                    []string
		matchExactly                 bool
	}{
		{name: "no filters"},
		{name: "plain names DB/SCH/TBL", catalog: sp("DB"), dbSchema: sp("SCH"), tableName: sp("TBL")},
		{name: "underscore names MY_DB/MY_SCH/MY_TBL", catalog: sp("MY_DB"), dbSchema: sp("MY_SCH"), tableName: sp("MY_TBL")},
		{name: "underscore catalog only MY_DB", catalog: sp("MY_DB")},
		{name: "underscore names matching two databases MULTI_DB/MY_SCH/MY_TBL", catalog: sp("MULTI_DB"), dbSchema: sp("MY_SCH"), tableName: sp("MY_TBL")},
		{name: "underscore names matching no database NO_DB/MY_SCH/MY_TBL", catalog: sp("NO_DB"), dbSchema: sp("MY_SCH"), tableName: sp("MY_TBL")},
		{name: "underscore names matching 20 databases MANY_DB/MY_SCH/MY_TBL", catalog: sp("MANY_DB"), dbSchema: sp("MY_SCH"), tableName: sp("MY_TBL")},
		{name: "percent patterns DB%/SCH%/TBL%", catalog: sp("DB%"), dbSchema: sp("SCH%"), tableName: sp("TBL%")},
		{name: "plain names with column filter COL", catalog: sp("DB"), dbSchema: sp("SCH"), tableName: sp("TBL"), columnName: sp("COL")},
		{name: "plain names with table type VIEW", catalog: sp("DB"), dbSchema: sp("SCH"), tableName: sp("TBL"), tableType: []string{"VIEW"}},
		{name: "plain names with unknown table type", catalog: sp("DB"), dbSchema: sp("SCH"), tableName: sp("TBL"), tableType: []string{"EXTERNAL"}},
		{name: "match exactly MY_DB/MY_SCH/MY_TBL", catalog: sp("MY_DB"), dbSchema: sp("MY_SCH"), tableName: sp("MY_TBL"), matchExactly: true},
		{name: "match exactly catalog only MY_DB", catalog: sp("MY_DB"), matchExactly: true},
		{name: "match exactly empty schema", catalog: sp("DB"), dbSchema: sp(""), matchExactly: true},
	}
	depths := []struct {
		file  string
		depth adbc.ObjectDepth
	}{
		{"catalogs.txt", adbc.ObjectDepthCatalogs},
		{"db_schemas.txt", adbc.ObjectDepthDBSchemas},
		{"tables.txt", adbc.ObjectDepthTables},
		{"all.txt", adbc.ObjectDepthAll},
	}

	for _, d := range depths {
		t.Run(strings.TrimSuffix(d.file, ".txt"), func(t *testing.T) {
			var out strings.Builder
			fmt.Fprintf(&out, "# Statements GetObjects sends at this depth, sorted. Regenerate with -update.\n")
			for _, in := range inputs {
				cnxn, fc := newFakeConnection(t)
				cnxn.matchExactly = in.matchExactly
				fc.Respond = fakeAccountShowTerse
				rdr, err := cnxn.GetObjects(context.Background(), d.depth, in.catalog, in.dbSchema, in.tableName, in.columnName, in.tableType)
				require.NoError(t, err, in.name)
				for rdr.Next() {
				}
				require.NoError(t, rdr.Err(), in.name)
				rdr.Release()

				plan := fc.Statements()
				lines := make([]string, len(plan))
				for i, st := range plan {
					lines[i] = describeStatement(st)
				}
				slices.Sort(lines)
				fmt.Fprintf(&out, "\n## %s: %d statement(s), %d IN ACCOUNT\n", in.name, len(lines), countInAccount(plan))
				for _, l := range lines {
					fmt.Fprintf(&out, "    %s\n", l)
				}
			}

			path := filepath.Join("testdata", "get_objects_query_plan", d.file)
			if *updateGolden {
				require.NoError(t, os.WriteFile(path, []byte(out.String()), 0o644))
				return
			}
			want, err := os.ReadFile(path)
			require.NoError(t, err, "missing golden file; run with -update")
			assert.Equal(t, string(want), out.String(), "GetObjects query plan changed; if intended, rerun with -update and review the diff")
		})
	}
}

var (
	reShowTerse           = regexp.MustCompile(`^SHOW TERSE /\* ADBC:getObjects \*/ (DATABASES|SCHEMAS)(?: LIKE '([^']*)')?`)
	reShowTerseStartsWith = regexp.MustCompile(` STARTS WITH '([^']*)'$`)
)

// fakeAccountShowTerse answers SHOW TERSE DATABASES/SCHEMAS queries the way an
// account would: no pattern matches two objects (OBJ1 and OBJ2), a pattern
// starting with MULTI matches two objects, one starting with MANY matches 20,
// one starting with NO_ matches none, and any other pattern matches one object
// named like the pattern. STARTS WITH 'X' (match_exactly) matches X and
// X_EXTRA. Everything else gets the default empty result.
func fakeAccountShowTerse(query string) (driver.Rows, error) {
	m := reShowTerse.FindStringSubmatch(query)
	if m == nil {
		return nil, nil
	}
	pattern := m[2]
	rows := &fakeRows{Cols: []string{"created_on", "name", "kind", "database_name", "schema_name"}}
	var names []string
	switch sw := reShowTerseStartsWith.FindStringSubmatch(query); {
	case sw != nil:
		names = []string{sw[1], sw[1] + "_EXTRA"}
	case pattern == "" || strings.Contains(pattern, "%"):
		names = []string{"OBJ1", "OBJ2"}
	case strings.HasPrefix(pattern, "MANY"):
		for i := range 20 {
			names = append(names, fmt.Sprintf("%s%02d", pattern, i))
		}
	case strings.HasPrefix(pattern, "MULTI"):
		names = []string{strings.Replace(pattern, "_", "-", 1), strings.Replace(pattern, "_", "X", 1)}
	case strings.HasPrefix(pattern, "NO_"):
	default:
		names = []string{pattern}
	}
	for _, n := range names {
		rows.Values = append(rows.Values, []driver.Value{"2026-01-01", n, "", "DB", ""})
	}
	return rows, nil
}

func countInAccount(plan []recordedStatement) int {
	n := 0
	for _, st := range plan {
		if strings.Contains(st.Query, " IN ACCOUNT") {
			n++
		}
	}
	return n
}

var _ driver.QueryerContext = (*fakeConn)(nil)
