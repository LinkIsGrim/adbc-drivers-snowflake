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
	"sync"
	"testing"
	"time"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/snowflakedb/gosnowflake/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConnectorConfigDoesNotMutateDatabase(t *testing.T) {
	db := &databaseImpl{
		cfg:                   &gosnowflake.Config{Params: map[string]*string{}},
		geographyOutputFormat: "GEOJSON",
		geometryOutputFormat:  "GEOJSON",
	}

	cfg := db.connectorConfig()
	require.NotNil(t, cfg.Params["GEOGRAPHY_OUTPUT_FORMAT"])
	assert.Equal(t, "GEOJSON", *cfg.Params["GEOGRAPHY_OUTPUT_FORMAT"])
	assert.Equal(t, "GEOJSON", *cfg.Params["GEOMETRY_OUTPUT_FORMAT"])
	assert.Empty(t, db.cfg.Params, "per-connection defaults leaked into the database config")

	// Changing the option after a connection was opened applies to the next one.
	require.NoError(t, db.SetOptionInternal(OptionGeographyOutputFormat, "EWKB", nil))
	cfg = db.connectorConfig()
	assert.Equal(t, "EWKB", *cfg.Params["GEOGRAPHY_OUTPUT_FORMAT"])
}

func TestConnectorConfigKeepsExplicitParams(t *testing.T) {
	db := &databaseImpl{
		cfg:                   &gosnowflake.Config{Params: map[string]*string{}},
		geographyOutputFormat: "GEOJSON",
		geometryOutputFormat:  "GEOJSON",
	}
	require.NoError(t, db.SetOptionInternal("GEOMETRY_OUTPUT_FORMAT", "WKT", nil))

	cfg := db.connectorConfig()
	assert.Equal(t, "WKT", *cfg.Params["GEOMETRY_OUTPUT_FORMAT"])
	assert.Equal(t, "GEOJSON", *cfg.Params["GEOGRAPHY_OUTPUT_FORMAT"])
	assert.Len(t, db.cfg.Params, 1)
}

// Opening connections concurrently from one database (as connection pools do)
// must not race on shared config. Run with -race. The host is a closed local
// port so that Connect fails fast without a Snowflake account.
func TestConcurrentOpen(t *testing.T) {
	drv := NewDriver(memory.DefaultAllocator)
	db, err := drv.NewDatabaseWithContext(context.Background(), map[string]string{
		adbc.OptionKeyURI: "user:pass@127.0.0.1:1/db?account=acct&protocol=http&loginTimeout=1",
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, db.Close(context.Background())) }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			cnxn, err := db.Open(ctx)
			if err == nil {
				_ = cnxn.Close(ctx)
			}
		})
	}
	wg.Wait()
}
