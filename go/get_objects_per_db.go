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
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/adbc-drivers/driverbase-go/driverbase"
	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/snowflakedb/gosnowflake/v2"
	"golang.org/x/sync/errgroup"
)

// GetObjects below Catalogs depth works one database at a time.
//
// The catalog pattern is first resolved to the matching databases with a
// single SHOW TERSE DATABASES, which only lists databases and is cheap even in
// large accounts. Each database is then queried on its own, scoped with IN
// DATABASE / IN SCHEMA, so no SHOW scans the whole account (see #205). At
// Columns depth this is also required for correctness: information_schema
// only covers the database it belongs to, so each catalog's columns, tables
// and schemas have to come from "<catalog>".information_schema. A literal
// catalog name is queried while SHOW TERSE DATABASES runs, and the result is
// kept only if the name matches the one database listed.
//
// With many matching databases, one account-wide SHOW per object type is
// cheaper than a scoped SHOW per database, so above
// getObjectsMaxScopedCatalogs the account-wide SHOWs are used instead: the
// SQL-template plan at DBSchemas and Tables depth, and SHOWs shared by every
// database at Columns depth.
const (
	getObjectsMaxScopedCatalogs = 16
	// getObjectsConcurrency bounds the databases queried at once.
	getObjectsConcurrency = 8
)

// getObjectsScope narrows the SHOW queries of the SQL-template plan. GetObjects
// only reuses the query ID of the SHOW TERSE DATABASES that resolved the
// catalogs; catalog and dbSchema stay nil, so the plan's SHOWs are
// account-wide.
type getObjectsScope struct {
	catalog, dbSchema        *string
	dbQueryID, schemaQueryID string
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

// matchesAll reports whether pattern filters nothing.
func matchesAll(pattern *string, matchExactly bool) bool {
	if pattern == nil {
		return true
	}
	return !matchExactly && (*pattern == "" || *pattern == "%" || *pattern == ".*")
}

// showTerseCatalogs lists the databases matching catalog with SHOW TERSE
// DATABASES and returns their names and the query ID.
func (c *connectionImpl) showTerseCatalogs(ctx context.Context, catalog *string) ([]string, string, error) {
	if hasEmptyLiteralFilter(c.matchExactly, catalog) {
		return nil, "", nil
	}
	names, queryID, err := showTerseNames(ctx, c.cn, buildShowTerseQuery(objDatabases, catalog, " IN ACCOUNT", c.matchExactly))
	if err != nil || !c.matchExactly || catalog == nil {
		return names, queryID, err
	}
	// STARTS WITH also matches longer names.
	return slices.DeleteFunc(names, func(name string) bool { return name != *catalog }), queryID, nil
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

// forEachCatalog calls fn for each catalog, getObjectsConcurrency at a time,
// and returns the results in catalog order.
func forEachCatalog(ctx context.Context, catalogs []string, fn func(ctx context.Context, catalog string) (driverbase.GetObjectsInfo, error)) ([]driverbase.GetObjectsInfo, error) {
	infos := make([]driverbase.GetObjectsInfo, len(catalogs))
	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(getObjectsConcurrency)
	for i, catalog := range catalogs {
		g.Go(func() (err error) {
			infos[i], err = fn(gCtx, catalog)
			return err
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return infos, nil
}

// constraintQueryIDs are the query IDs of the SHOW queries that
// get_objects_all.sql reads through RESULT_SCAN.
type constraintQueryIDs struct {
	pk, fk, unique, columns string
}

// Typed empty sources for RESULT_SCAN, with the columns get_objects_all.sql
// reads from each SHOW.
const (
	emptyPkUkQuery    = `SELECT NULL::VARCHAR AS "database_name", NULL::VARCHAR AS "schema_name", NULL::VARCHAR AS "table_name", NULL::VARCHAR AS "constraint_name", NULL::VARCHAR AS "column_name", NULL::NUMBER AS "key_sequence" WHERE FALSE`
	emptyFkQuery      = `SELECT NULL::VARCHAR AS "fk_database_name", NULL::VARCHAR AS "fk_schema_name", NULL::VARCHAR AS "fk_table_name", NULL::VARCHAR AS "fk_name", NULL::VARCHAR AS "fk_column_name", NULL::VARCHAR AS "pk_database_name", NULL::VARCHAR AS "pk_schema_name", NULL::VARCHAR AS "pk_table_name", NULL::VARCHAR AS "pk_column_name", NULL::NUMBER AS "key_sequence" WHERE FALSE`
	emptyColumnsQuery = `SELECT NULL::VARCHAR AS "database_name", NULL::VARCHAR AS "schema_name", NULL::VARCHAR AS "table_name", NULL::VARCHAR AS "column_name", NULL::VARCHAR AS "data_type" WHERE FALSE`
)

// showConstraints runs the SHOW queries for constraints and column details
// concurrently with the given IN ... scope. columnsScope may narrow SHOW
// COLUMNS further, to a single table. A scope that does not exist, and a
// filter that can match nothing, give empty results.
func (c *connectionImpl) showConstraints(ctx context.Context, scope, columnsScope string, dbSchema, tableName, columnName *string) (ids constraintQueryIDs, err error) {
	noKeys := hasEmptyLiteralFilter(c.matchExactly, dbSchema, tableName)
	noColumns := noKeys || hasEmptyLiteralFilter(c.matchExactly, columnName)
	show := func(ctx context.Context, query, emptyQuery string, empty bool) (string, error) {
		if empty {
			query = emptyQuery
		}
		return getQueryID(ctx, query, c.cn, emptyQuery)
	}

	g, gCtx := errgroup.WithContext(ctx)
	g.Go(func() (err error) {
		ids.pk, err = show(gCtx, "SHOW PRIMARY KEYS /* ADBC:getObjectsTables */"+scope, emptyPkUkQuery, noKeys)
		return err
	})
	g.Go(func() (err error) {
		ids.fk, err = show(gCtx, "SHOW IMPORTED KEYS /* ADBC:getObjectsTables */"+scope, emptyFkQuery, noKeys)
		return err
	})
	g.Go(func() (err error) {
		ids.unique, err = show(gCtx, "SHOW UNIQUE KEYS /* ADBC:getObjectsTables */"+scope, emptyPkUkQuery, noKeys)
		return err
	})
	g.Go(func() (err error) {
		ids.columns, err = show(gCtx, "SHOW COLUMNS /* ADBC:getObjects */"+columnsScope, emptyColumnsQuery, noColumns)
		return err
	})
	return ids, g.Wait()
}

// catalogResult is the result of querying one database.
type catalogResult struct {
	info driverbase.GetObjectsInfo
	err  error
}

// getObjectsColumns builds the Columns-depth result for more than
// getObjectsMaxScopedCatalogs databases, with constraint SHOWs shared by all
// of them.
func (c *connectionImpl) getObjectsColumns(ctx context.Context, catalogs []string, dbSchema, tableName, columnName *string) ([]driverbase.GetObjectsInfo, error) {
	shared, err := c.showConstraints(ctx, " IN ACCOUNT", " IN ACCOUNT", dbSchema, tableName, columnName)
	if err != nil {
		return nil, err
	}
	return forEachCatalog(ctx, catalogs, func(ctx context.Context, catalog string) (driverbase.GetObjectsInfo, error) {
		return c.getObjectsColumnsIn(ctx, catalog, &shared, dbSchema, tableName, columnName)
	})
}

// getObjectsColumnsIn builds the Columns-depth result for one database. Unless
// shared constraint SHOWs are given, they are run scoped to the database, or
// to the schema when the schema filter matches one.
func (c *connectionImpl) getObjectsColumnsIn(ctx context.Context, catalog string, shared *constraintQueryIDs, dbSchema, tableName, columnName *string) (driverbase.GetObjectsInfo, error) {
	ids := shared
	if ids == nil {
		scope := " IN DATABASE " + quoteIdentifier(catalog)
		columnsScope := scope
		if !matchesAll(dbSchema, c.matchExactly) {
			schemas, err := c.execShowSchemas(ctx, dbSchema, scope)
			if err != nil {
				return driverbase.GetObjectsInfo{}, err
			}
			switch len(schemas) {
			case 0:
				return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: []driverbase.DBSchemaInfo{}}, nil
			case 1:
				schema := quoteIdentifier(catalog) + "." + quoteIdentifier(schemas[0].schemaName)
				scope = " IN SCHEMA " + schema
				columnsScope = scope
				table, ok := scopeIdentifier(tableName)
				if c.matchExactly {
					ok = tableName != nil && *tableName != ""
					if ok {
						table = *tableName
					}
				}
				if ok {
					columnsScope = " IN TABLE " + schema + "." + quoteIdentifier(table)
				}
			}
		}
		got, err := c.showConstraints(ctx, scope, columnsScope, dbSchema, tableName, columnName)
		if err != nil {
			return driverbase.GetObjectsInfo{}, err
		}
		ids = &got
	}
	return c.queryCatalogColumns(ctx, catalog, *ids, dbSchema, tableName, columnName)
}

// queryCatalogColumns runs get_objects_all.sql against one database.
func (c *connectionImpl) queryCatalogColumns(ctx context.Context, catalog string, ids constraintQueryIDs, dbSchema, tableName, columnName *string) (info driverbase.GetObjectsInfo, err error) {
	args := []sql.NamedArg{
		sql.Named("DB_NAME", catalog),
		metadataPatternArg("CATALOG", nil, c.matchExactly),
		metadataPatternArg("DB_SCHEMA", dbSchema, c.matchExactly),
		metadataPatternArg("TABLE", tableName, c.matchExactly),
		metadataPatternArg("COLUMN", columnName, c.matchExactly),
		sql.Named("PK_QUERY_ID", ids.pk),
		sql.Named("FK_QUERY_ID", ids.fk),
		sql.Named("UNIQUE_QUERY_ID", ids.unique),
		sql.Named("SHOW_COLUMNS_QUERY_ID", ids.columns),
	}
	nvargs := make([]driver.NamedValue, len(args))
	for i, arg := range args {
		nvargs[i] = driver.NamedValue{Name: arg.Name, Ordinal: i + 1, Value: arg.Value}
	}

	query := fmt.Sprintf(queryGetObjectsAll, quoteIdentifier(catalog))
	if c.matchExactly {
		query = strings.ReplaceAll(query, " ILIKE ", " LIKE ")
	}
	rows, err := c.cn.QueryContext(ctx, query, nvargs)
	if err != nil {
		if isMetadataNotFound(err) {
			// The database was dropped since it was listed, or its
			// information_schema is not readable with the current role.
			return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: []driverbase.DBSchemaInfo{}}, nil
		}
		return info, errToAdbcErr(adbc.StatusIO, err)
	}
	defer func() {
		err = errors.Join(err, rows.Close())
	}()

	// The query aggregates to exactly one row; treat a missing one as a
	// catalog with no matching schemas.
	dest := make([]driver.Value, len(rows.Columns()))
	if err := rows.Next(dest); err != nil {
		if errors.Is(err, io.EOF) {
			return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: []driverbase.DBSchemaInfo{}}, nil
		}
		return info, errToAdbcErr(adbc.StatusInvalidData, err)
	}
	if err := info.Scan(dest[0]); err != nil {
		return info, errToAdbcErr(adbc.StatusInvalidData, err)
	}

	// A few columns need additional processing outside of Snowflake
	for i, sch := range info.CatalogDbSchemas {
		for j, tab := range sch.DbSchemaTables {
			for k, col := range tab.TableColumns {
				field := c.toArrowField(col)
				xdbcDataType := driverbase.ToXdbcDataType(field.Type)
				if field.Type != nil {
					info.CatalogDbSchemas[i].DbSchemaTables[j].TableColumns[k].XdbcDataType = new(int16(field.Type.ID()))
				}
				info.CatalogDbSchemas[i].DbSchemaTables[j].TableColumns[k].XdbcSqlDataType = new(int16(xdbcDataType))
			}
		}
	}
	return info, nil
}
