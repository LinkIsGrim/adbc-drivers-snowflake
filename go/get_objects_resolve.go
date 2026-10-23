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
	"strings"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/snowflakedb/gosnowflake/v2"
)

// getObjectsScope records how narrowly GetObjects can scope its SHOW queries.
//
// The catalog and schema arguments are ADBC patterns, where '_' matches any
// single character, so a name such as MY_DB can't be used for an IN DATABASE
// scope directly. Without anything else to go on, every SHOW that can't be
// scoped falls back to IN ACCOUNT, which scans the whole account (see #205).
// Instead, a pattern that contains '_' but no '%' is resolved first with the
// SHOW TERSE query the plan needs anyway: if it matches exactly one object,
// that object's name is used for scoping. The patterns themselves are still
// what results are filtered with, so the result is unchanged.
type getObjectsScope struct {
	// catalog and dbSchema are concrete names to scope by, set only when the
	// corresponding pattern is a literal name or resolved to one object.
	catalog, dbSchema *string

	// dbQueryID and schemaQueryID are the query IDs of the resolution
	// queries, which are the same SHOW TERSE DATABASES / SCHEMAS queries
	// the SQL templates read through RESULT_SCAN.
	dbQueryID, schemaQueryID string
}

// isResolvablePattern reports whether pattern is worth resolving: it has
// single-character wildcards but no '%', so it can match only a few names
// and almost always names one object.
func isResolvablePattern(pattern *string) bool {
	return pattern != nil && strings.Contains(*pattern, "_") && !strings.Contains(*pattern, "%")
}

// literalName returns pattern as a concrete name if it has no wildcards.
func literalName(pattern *string) *string {
	if pattern == nil || isWildcardStr(*pattern) {
		return nil
	}
	return pattern
}

// literalScope is the scope the catalog and dbSchema filters give without
// resolving anything. With match_exactly they are names already; an empty
// name matches nothing, which the SHOW helpers check for themselves.
func literalScope(catalog, dbSchema *string, matchExactly bool) getObjectsScope {
	if !matchExactly {
		return getObjectsScope{catalog: literalName(catalog), dbSchema: literalName(dbSchema)}
	}
	scope := getObjectsScope{catalog: catalog}
	if dbSchema != nil && *dbSchema != "" {
		scope.dbSchema = dbSchema
	}
	return scope
}

// resolveGetObjectsScope works out the scope for a GetObjects call. empty is
// true when the catalog pattern matches no database, in which case the result
// is empty and nothing else needs to run.
func resolveGetObjectsScope(ctx context.Context, conn driver.QueryerContext, depth adbc.ObjectDepth, catalog, dbSchema *string, matchExactly bool) (scope getObjectsScope, empty bool, err error) {
	scope = literalScope(catalog, dbSchema, matchExactly)
	if depth == adbc.ObjectDepthCatalogs || matchExactly {
		// A single SHOW TERSE DATABASES is the whole plan, or the filters
		// are names already.
		return scope, false, nil
	}

	if scope.catalog == nil && isResolvablePattern(catalog) {
		names, queryID, err := showTerseNames(ctx, conn, buildShowTerseQuery(objDatabases, catalog, " IN ACCOUNT", false))
		if err != nil {
			return scope, false, err
		}
		switch len(names) {
		case 0:
			return scope, true, nil
		case 1:
			scope.catalog = &names[0]
		}
		scope.dbQueryID = queryID
	}

	// Schema patterns only narrow the scope at Tables depth and below. At
	// DBSchemas depth the schema list is the result itself.
	if depth != adbc.ObjectDepthDBSchemas && scope.catalog != nil && scope.dbSchema == nil && isResolvablePattern(dbSchema) {
		names, queryID, err := showTerseNames(ctx, conn, buildShowTerseQuery(objSchemas, dbSchema, " IN DATABASE "+quoteIdentifier(*scope.catalog), false))
		if err != nil {
			return scope, false, err
		}
		if len(names) == 1 {
			scope.dbSchema = &names[0]
		}
		scope.schemaQueryID = queryID
	}
	return scope, false, nil
}

// showTerseNames runs a SHOW TERSE query and returns the "name" column and the
// query ID. A SHOW that matches nothing yields no names and no query ID.
func showTerseNames(ctx context.Context, conn driver.QueryerContext, query string) (names []string, queryID string, err error) {
	rows, err := conn.QueryContext(ctx, query, nil)
	if err != nil {
		if isMetadataNotFound(err) {
			return nil, "", nil
		}
		return nil, "", errToAdbcErr(adbc.StatusIO, err)
	}
	defer func() {
		err = errors.Join(err, rows.Close())
	}()

	cols := rows.Columns()
	nameIdx := -1
	for i, col := range cols {
		if col == "name" {
			nameIdx = i
		}
	}
	dest := make([]driver.Value, len(cols))
	for {
		if err := rows.Next(dest); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, "", errToAdbcErr(adbc.StatusIO, err)
		}
		if nameIdx >= 0 {
			if name, ok := dest[nameIdx].(string); ok {
				names = append(names, name)
			}
		}
	}
	if sfRows, ok := rows.(gosnowflake.SnowflakeRows); ok {
		queryID = sfRows.GetQueryID()
	}
	return names, queryID, nil
}
