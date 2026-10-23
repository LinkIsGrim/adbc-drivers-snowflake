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

	"github.com/adbc-drivers/driverbase-go/driverbase"
	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"golang.org/x/sync/errgroup"
)

// This file reads SHOW TERSE results directly, instead of going through
// RESULT_SCAN SQL templates, to build one catalog of a DBSchemas- or
// Tables-depth GetObjects result.

// getObjectsDBSchemasIn returns the schemas of catalog that match dbSchema,
// with 1 SHOW query.
func (c *connectionImpl) getObjectsDBSchemasIn(ctx context.Context, catalog string, dbSchema *string) (driverbase.GetObjectsInfo, error) {
	schemas, err := c.execShowSchemas(ctx, dbSchema, " IN DATABASE "+quoteTblName(catalog))
	if err != nil {
		return driverbase.GetObjectsInfo{}, err
	}

	dbSchemas := make([]driverbase.DBSchemaInfo, 0, len(schemas))
	for _, s := range schemas {
		sn := s.schemaName
		dbSchemas = append(dbSchemas, driverbase.DBSchemaInfo{
			DbSchemaName: &sn,
		})
	}
	return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: dbSchemas}, nil
}

// getObjectsTablesIn returns the schemas of catalog that match dbSchema, with
// their tables that match tableName.
//   - No schema filter: 2 parallel SHOW queries over the database.
//   - A schema filter that matches one schema: SHOW SCHEMAS, then SHOW
//     TABLES in that schema. A literal schema name runs both at once.
//   - A schema filter that matches several: SHOW SCHEMAS, then SHOW TABLES
//     over the database.
func (c *connectionImpl) getObjectsTablesIn(ctx context.Context, catalog string, dbSchema, tableName *string, objType string) (driverbase.GetObjectsInfo, error) {
	var (
		schemas      []schemaEntry
		tableEntries []tableEntry
	)
	escapedCatalog := quoteTblName(catalog)
	dbSuffix := " IN DATABASE " + escapedCatalog

	if matchesAll(dbSchema, c.matchExactly) {
		g, gCtx := errgroup.WithContext(ctx)
		g.Go(func() (err error) {
			schemas, err = c.execShowSchemas(gCtx, dbSchema, dbSuffix)
			return err
		})
		g.Go(func() (err error) {
			tableEntries, err = c.execShowTables(gCtx, objType, tableName, dbSuffix)
			return err
		})
		if err := g.Wait(); err != nil {
			return driverbase.GetObjectsInfo{}, err
		}
	} else {
		// A literal schema filter is usually the schema's exact name, so list
		// that schema's tables while SHOW SCHEMAS confirms it.
		type showResult struct {
			entries []tableEntry
			err     error
		}
		var guessed chan showResult
		guess := literalScope(nil, dbSchema, c.matchExactly).dbSchema
		if guess != nil {
			guessCtx, cancelGuess := context.WithCancel(ctx)
			defer cancelGuess()
			guessed = make(chan showResult, 1)
			go func() {
				entries, err := c.execShowTables(guessCtx, objType, tableName, " IN SCHEMA "+escapedCatalog+"."+quoteTblName(*guess))
				guessed <- showResult{entries, err}
			}()
		}

		var err error
		if schemas, err = c.execShowSchemas(ctx, dbSchema, dbSuffix); err != nil {
			return driverbase.GetObjectsInfo{}, err
		}
		switch {
		case len(schemas) == 0:
			return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: []driverbase.DBSchemaInfo{}}, nil
		case len(schemas) == 1 && guess != nil && schemas[0].schemaName == *guess:
			r := <-guessed
			if r.err != nil {
				return driverbase.GetObjectsInfo{}, r.err
			}
			tableEntries = r.entries
		default:
			suffix := dbSuffix
			if len(schemas) == 1 {
				suffix = " IN SCHEMA " + escapedCatalog + "." + quoteTblName(schemas[0].schemaName)
			}
			if tableEntries, err = c.execShowTables(ctx, objType, tableName, suffix); err != nil {
				return driverbase.GetObjectsInfo{}, err
			}
		}
	}

	// Group tables by schema, only including tables in matching schemas
	tablesBySchema := make(map[string][]driverbase.TableInfo, len(schemas))
	for _, s := range schemas {
		tablesBySchema[s.schemaName] = []driverbase.TableInfo{}
	}
	for _, e := range tableEntries {
		if tables, ok := tablesBySchema[e.schemaName]; ok {
			tablesBySchema[e.schemaName] = append(tables, driverbase.TableInfo{
				TableName: e.tableName,
				TableType: e.tableType,
			})
		}
	}

	// Build DBSchemaInfo list preserving schema order from SHOW
	dbSchemas := make([]driverbase.DBSchemaInfo, 0, len(schemas))
	for _, s := range schemas {
		sn := s.schemaName
		dbSchemas = append(dbSchemas, driverbase.DBSchemaInfo{
			DbSchemaName:   &sn,
			DbSchemaTables: tablesBySchema[sn],
		})
	}
	return driverbase.GetObjectsInfo{CatalogName: &catalog, CatalogDbSchemas: dbSchemas}, nil
}

type tableEntry struct {
	dbName     string
	schemaName string
	tableName  string
	tableType  string
}

type schemaEntry struct {
	dbName     string
	schemaName string
}

// showObjType returns the SHOW object type string based on the tableType filter.
func showObjType(tableType []string) string {
	if len(tableType) == 1 {
		if strings.EqualFold("VIEW", tableType[0]) {
			return objViews
		}
		if strings.EqualFold("TABLE", tableType[0]) {
			return objTables
		}
	}
	return objObjects
}

// execShowTables executes a SHOW TERSE command for tables/objects/views
// and reads the results directly into a slice.
func (c *connectionImpl) execShowTables(ctx context.Context, objType string, pattern *string, suffix string) (entries []tableEntry, err error) {
	if hasEmptyLiteralFilter(c.matchExactly, pattern) {
		return nil, nil
	}
	query := buildShowTerseQuery(objType, pattern, suffix, c.matchExactly)
	rows, err := c.cn.QueryContext(ctx, query, nil)
	if err != nil {
		if isMetadataNotFound(err) {
			return nil, nil
		}
		return nil, errToAdbcErr(adbc.StatusIO, err)
	}
	defer func() {
		err = errors.Join(err, rows.Close())
	}()

	entries, err = readTableEntries(rows)
	if err != nil {
		return nil, err
	}
	if c.matchExactly && pattern != nil {
		matched := entries[:0]
		for _, entry := range entries {
			if entry.tableName == *pattern {
				matched = append(matched, entry)
			}
		}
		entries = matched
	}
	return entries, nil
}

func readTableEntries(rows driver.Rows) ([]tableEntry, error) {
	cols := rows.Columns()
	nameIdx, kindIdx, dbIdx, schIdx := -1, -1, -1, -1
	for i, col := range cols {
		switch col {
		case "name":
			nameIdx = i
		case "kind":
			kindIdx = i
		case "database_name":
			dbIdx = i
		case "schema_name":
			schIdx = i
		}
	}

	dest := make([]driver.Value, len(cols))
	var entries []tableEntry
	for {
		if err := rows.Next(dest); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, errToAdbcErr(adbc.StatusIO, err)
		}

		entry := tableEntry{}
		if nameIdx >= 0 {
			entry.tableName = dest[nameIdx].(string)
		}
		if kindIdx >= 0 {
			entry.tableType = dest[kindIdx].(string)
		}
		if dbIdx >= 0 {
			entry.dbName = dest[dbIdx].(string)
		}
		if schIdx >= 0 {
			entry.schemaName = dest[schIdx].(string)
		}
		entries = append(entries, entry)
	}

	return entries, nil
}

// execShowSchemas executes a SHOW TERSE SCHEMAS command and reads the results directly.
func (c *connectionImpl) execShowSchemas(ctx context.Context, pattern *string, suffix string) (_ []schemaEntry, err error) {
	if hasEmptyLiteralFilter(c.matchExactly, pattern) {
		return nil, nil
	}
	query := buildShowTerseQuery(objSchemas, pattern, suffix, c.matchExactly)
	rows, err := c.cn.QueryContext(ctx, query, nil)
	if err != nil {
		// Missing or inaccessible scopes contribute no schemas.
		if isMetadataNotFound(err) {
			return nil, nil
		}
		return nil, errToAdbcErr(adbc.StatusIO, err)
	}
	defer func() {
		err = errors.Join(err, rows.Close())
	}()

	// SHOW TERSE is not actually fixed and has changed between versions in the past,
	// so we look up by name to avoid possible breaking if snowflake adds, removes,
	// or reorders columns in the future.
	cols := rows.Columns()
	nameIdx, dbIdx := -1, -1
	for i, col := range cols {
		switch col {
		case "name":
			nameIdx = i
		case "database_name":
			dbIdx = i
		}
	}

	dest := make([]driver.Value, len(cols))
	var entries []schemaEntry
	for {
		if err := rows.Next(dest); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, errToAdbcErr(adbc.StatusIO, err)
		}

		entry := schemaEntry{}
		if nameIdx >= 0 {
			entry.schemaName = dest[nameIdx].(string)
		}
		if dbIdx >= 0 {
			entry.dbName = dest[dbIdx].(string)
		}
		if !c.matchExactly || pattern == nil || entry.schemaName == *pattern {
			entries = append(entries, entry)
		}
	}

	return entries, nil
}

// buildGetObjectsResult creates a RecordReader from GetObjectsInfo values
// using the standard driverbase builder.
func buildGetObjectsResult(alloc memory.Allocator, infos ...driverbase.GetObjectsInfo) (array.RecordReader, error) {
	ch := make(chan driverbase.GetObjectsInfo, len(infos)+1)
	errCh := make(chan error)
	for _, info := range infos {
		ch <- info
	}
	close(ch)
	return driverbase.BuildGetObjectsRecordReader(alloc, ch, errCh)
}
