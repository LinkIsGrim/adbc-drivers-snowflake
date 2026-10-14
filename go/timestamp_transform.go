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
	"math"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/snowflakedb/gosnowflake/v2"
)

// timestampConverter turns Snowflake's timestamp representations into Arrow
// timestamps of the result column's unit using integer arithmetic.
//
// The result is always the instant floored to the output unit, which is what
// building a time.Time and passing it through getArrowTimestampFromTime
// produces. Values near the edges of the representable range, where that path
// either reports an error or wraps, are left to a fallback that runs the
// time.Time path for that value, so results and error messages stay identical.
type timestampConverter struct {
	// Compact values scaled by 10^scale become scaled*mul/div; exactly one of
	// mul and div is not 1. Values outside [minScaled, maxScaled] take the
	// fallback.
	mul, div             int64
	minScaled, maxScaled int64

	// Struct values (epoch seconds, nanoseconds) become
	// epoch*unitsPerSecond + nanoseconds/nanosPerUnit.
	unitsPerSecond, nanosPerUnit int64
	// checkNanoOverflow mirrors the NanosecondsNoOverflow check in
	// getArrowTimestampFromTime.
	checkNanoOverflow bool

	// scaleDivisor and fractionMultiplier rebuild a time.Time from a compact
	// value for the fallback.
	scaleDivisor, fractionMultiplier int64
}

// The NanosecondsNoOverflow check compares whole seconds against
// math.MaxInt64/1e9 and math.MinInt64/1e9 as floats, so it rejects these
// second values and beyond.
const (
	noOverflowMaxSeconds = math.MaxInt64/1_000_000_000 + 1
	noOverflowMinSeconds = math.MinInt64/1_000_000_000 - 1
)

func newTimestampConverter(scale int64, unit, originalArrowUnit arrow.TimeUnit, maxTimestampPrecision MaxTimestampPrecision) timestampConverter {
	c := timestampConverter{
		mul: 1, div: 1,
		scaleDivisor:       int64(math.Pow10(int(scale))),
		fractionMultiplier: int64(math.Pow10(9 - int(scale))),
		minScaled:          math.MinInt64, maxScaled: math.MaxInt64,
		checkNanoOverflow: maxTimestampPrecision == NanosecondsNoOverflow && originalArrowUnit == arrow.Nanosecond,
	}
	if maxTimestampPrecision == Microseconds && originalArrowUnit == arrow.Nanosecond {
		unit = arrow.Microsecond
	}
	c.unitsPerSecond = int64(math.Pow10(3 * int(unit)))
	c.nanosPerUnit = 1_000_000_000 / c.unitsPerSecond

	pow := int64(math.Pow10(int(scale)))
	if digits := 3*int64(unit) - scale; digits >= 0 {
		c.mul = int64(math.Pow10(int(digits)))
		c.minScaled, c.maxScaled = math.MinInt64/c.mul, math.MaxInt64/c.mul
	} else {
		// Only when Microseconds truncates a column with more digits; the
		// result always fits.
		c.div = int64(math.Pow10(int(-digits)))
	}
	if c.checkNanoOverflow {
		// floor(scaled / 10^scale) must stay strictly inside the bounds.
		if noOverflowMaxSeconds <= math.MaxInt64/pow {
			c.maxScaled = min(c.maxScaled, noOverflowMaxSeconds*pow-1)
		}
		if noOverflowMinSeconds+1 >= math.MinInt64/pow {
			c.minScaled = max(c.minScaled, (noOverflowMinSeconds+1)*pow)
		}
	}
	return c
}

func floorDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// fromScaled converts a compact value scaled by 10^scale.
func (c *timestampConverter) fromScaled(v int64) (arrow.Timestamp, bool) {
	if v < c.minScaled || v > c.maxScaled {
		return 0, false
	}
	if c.div != 1 {
		return arrow.Timestamp(floorDiv(v, c.div)), true
	}
	return arrow.Timestamp(v * c.mul), true
}

// fromEpochFraction converts the (epoch seconds, nanoseconds) struct
// representation. Fractions outside [0, 1e9), which time.Unix would
// normalize, take the fallback.
func (c *timestampConverter) fromEpochFraction(epoch int64, fraction int32) (arrow.Timestamp, bool) {
	if fraction < 0 || fraction >= 1e9 {
		return 0, false
	}
	if c.checkNanoOverflow && (epoch >= noOverflowMaxSeconds || epoch <= noOverflowMinSeconds) {
		return 0, false
	}
	if epoch > math.MaxInt64/c.unitsPerSecond || epoch < math.MinInt64/c.unitsPerSecond {
		return 0, false
	}
	v := epoch * c.unitsPerSecond
	frac := int64(fraction) / c.nanosPerUnit
	if v > math.MaxInt64-frac {
		return 0, false
	}
	return arrow.Timestamp(v + frac), true
}

// isIdentity reports whether compact values are already timestamps of the
// output unit, as when a column's scale is 3, 6 or 9 and matches it.
func (c *timestampConverter) isIdentity() bool {
	return c.mul == 1 && c.div == 1
}

// inRange reports whether every non-null value in scaled converts without the
// fallback. a supplies the validity bitmap.
func (c *timestampConverter) inRange(a arrow.Array, scaled []int64) bool {
	if c.minScaled == math.MinInt64 && c.maxScaled == math.MaxInt64 {
		return true
	}
	for i, v := range scaled {
		if (v < c.minScaled || v > c.maxScaled) && a.IsValid(i) {
			return false
		}
	}
	return true
}

// retypeTimestamps returns an array of type dt that shares validity and values
// with offset, without copying. Arrow timestamps are int64 values, so this is
// the whole conversion when the converter is the identity.
func retypeTimestamps(dt *arrow.TimestampType, n int, validity, values *memory.Buffer, nulls, offset int) arrow.Array {
	data := array.NewData(dt, n, []*memory.Buffer{validity, values}, nil, nulls, offset)
	defer data.Release()
	return array.MakeFromData(data)
}

// newTimestampArray allocates the output for a: a values buffer of a.Len()
// timestamps and a copy of a's validity bitmap.
func newTimestampArray(ctx context.Context, a arrow.Array) (values, validity *memory.Buffer, out []arrow.Timestamp) {
	pool := compute.GetAllocator(ctx)
	n := a.Len()
	values = memory.NewResizableBuffer(pool)
	values.Resize(arrow.TimestampTraits.BytesRequired(n))
	if a.NullN() > 0 {
		validity = memory.NewResizableBuffer(pool)
		validity.Resize(int(bitutil.BytesForBits(int64(n))))
		bitutil.CopyBitmap(a.NullBitmapBytes(), a.Data().Offset(), n, validity.Bytes(), 0)
	}
	return values, validity, arrow.TimestampTraits.CastFromBytes(values.Bytes())
}

// finishTimestampArray wraps the buffers from newTimestampArray, or releases
// them if err is set.
func finishTimestampArray(dt *arrow.TimestampType, a arrow.Array, values, validity *memory.Buffer, err error) (arrow.Array, error) {
	defer values.Release()
	if validity != nil {
		defer validity.Release()
	}
	if err != nil {
		return nil, err
	}
	data := array.NewData(dt, a.Len(), []*memory.Buffer{validity, values}, nil, a.NullN(), 0)
	defer data.Release()
	return array.MakeFromData(data), nil
}

// timestampTransformer returns the column transformer for TIMESTAMP_NTZ,
// TIMESTAMP_LTZ and TIMESTAMP_TZ results. withOffset is set for TIMESTAMP_TZ,
// whose struct representations carry a time zone offset. loc is the session
// time zone for TIMESTAMP_LTZ and nil otherwise. Neither changes the instant;
// they only affect how a value is rendered in an overflow error.
func timestampTransformer(withOffset bool, dt *arrow.TimestampType, scale int64, originalArrowUnit arrow.TimeUnit, maxTimestampPrecision MaxTimestampPrecision, loc *time.Location) colTransformer {
	conv := newTimestampConverter(scale, dt.Unit, originalArrowUnit, maxTimestampPrecision)
	toArrow := func(t time.Time) (arrow.Timestamp, error) {
		return getArrowTimestampFromTime(t, dt.Unit, originalArrowUnit, maxTimestampPrecision)
	}
	inLoc := func(t time.Time, l *time.Location) time.Time {
		if l == nil {
			return t
		}
		return t.In(l)
	}

	// convertScaled and convertEpochFraction fill out for every slot of a,
	// calling fallback for the non-null values the integer path leaves to it.
	// Null slots are converted too, which is cheaper than testing each one;
	// whatever they hold is masked by the validity bitmap.
	convertScaled := func(a arrow.Array, out []arrow.Timestamp, scaled []int64, fallback func(i int) (arrow.Timestamp, error)) error {
		for i, v := range scaled {
			ts, ok := conv.fromScaled(v)
			if !ok && a.IsValid(i) {
				var err error
				if ts, err = fallback(i); err != nil {
					return err
				}
			}
			out[i] = ts
		}
		return nil
	}
	convertEpochFraction := func(a arrow.Array, out []arrow.Timestamp, epoch []int64, fraction []int32, fallback func(i int) (arrow.Timestamp, error)) error {
		for i, e := range epoch {
			ts, ok := conv.fromEpochFraction(e, fraction[i])
			if !ok && a.IsValid(i) {
				var err error
				if ts, err = fallback(i); err != nil {
					return err
				}
			}
			out[i] = ts
		}
		return nil
	}

	return func(ctx context.Context, a arrow.Array) (arrow.Array, error) {
		if a.DataType().ID() != arrow.STRUCT {
			// Compact form: one int64 scaled by 10^scale.
			scaled := a.(*array.Int64).Int64Values()
			if conv.isIdentity() && conv.inRange(a, scaled) {
				buffers := a.Data().Buffers()
				return retypeTimestamps(dt, a.Len(), buffers[0], buffers[1], a.NullN(), a.Data().Offset()), nil
			}
			values, validity, out := newTimestampArray(ctx, a)
			err := convertScaled(a, out, scaled, func(i int) (arrow.Timestamp, error) {
				return toArrow(inLoc(scaledTimestampToTime(scaled[i], conv.scaleDivisor, conv.fractionMultiplier), loc))
			})
			return finishTimestampArray(dt, a, values, validity, err)
		}

		structData := a.(*array.Struct)
		if withOffset && structData.NumField() == 2 {
			// Compact TIMESTAMP_TZ: scaled int64 plus a time zone offset.
			scaledArr := structData.Field(0).(*array.Int64)
			scaled := scaledArr.Int64Values()
			tzoffset := structData.Field(1).(*array.Int32).Int32Values()
			// The struct's validity and the child's values can be combined
			// as is only when both start at the same offset.
			if conv.isIdentity() && scaledArr.Data().Offset() == a.Data().Offset() && conv.inRange(a, scaled) {
				return retypeTimestamps(dt, a.Len(), a.Data().Buffers()[0], scaledArr.Data().Buffers()[1], a.NullN(), a.Data().Offset()), nil
			}
			values, validity, out := newTimestampArray(ctx, a)
			err := convertScaled(a, out, scaled, func(i int) (arrow.Timestamp, error) {
				l := snowflakeOffsetLocation(tzoffset[i])
				return toArrow(scaledTimestampToTime(scaled[i], conv.scaleDivisor, conv.fractionMultiplier).In(l))
			})
			return finishTimestampArray(dt, a, values, validity, err)
		}

		// Wide form: epoch seconds plus nanoseconds (plus an offset for TIMESTAMP_TZ).
		values, validity, out := newTimestampArray(ctx, a)
		epoch := structData.Field(0).(*array.Int64).Int64Values()
		fraction := structData.Field(1).(*array.Int32).Int32Values()
		var tzoffset []int32
		if withOffset {
			tzoffset = structData.Field(2).(*array.Int32).Int32Values()
		}
		err := convertEpochFraction(a, out, epoch, fraction, func(i int) (arrow.Timestamp, error) {
			t := time.Unix(epoch[i], int64(fraction[i]))
			if tzoffset != nil {
				t = t.In(snowflakeOffsetLocation(tzoffset[i]))
			}
			return toArrow(t)
		})
		return finishTimestampArray(dt, a, values, validity, err)
	}
}

// snowflakeOffsetLocation returns the fixed zone for a TIMESTAMP_TZ offset,
// which Snowflake encodes as minutes from UTC plus 1440.
func snowflakeOffsetLocation(offset int32) *time.Location {
	return gosnowflake.Location(int(offset) - 1440)
}
