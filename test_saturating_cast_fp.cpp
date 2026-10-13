// test_saturating_cast_fp.cpp
//
// Tests for the reference implementation in saturating_cast_fp.hpp.
// Compile-time behaviour is checked with static_assert; runtime edge
// cases with assert. Prints PASS at the end if everything succeeded.

#include "saturating_cast_fp.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using std_proposal::saturating_cast;

namespace {
constexpr double kNaN  = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf  = std::numeric_limits<double>::infinity();
} // namespace

// ---------------------------------------------------------------------------
// Compile-time tests: the function must be usable in constant expressions.
// ---------------------------------------------------------------------------

// In-range truncation toward zero, both signs.
static_assert(saturating_cast<int>(3.99) == 3);
static_assert(saturating_cast<int>(-3.99) == -3);
static_assert(saturating_cast<int>(0.0) == 0);
static_assert(saturating_cast<int>(-0.0) == 0);
static_assert(saturating_cast<unsigned>(42.7) == 42u);

// Saturation on overflow / underflow, 32-bit.
static_assert(saturating_cast<int>(1.0e30) == std::numeric_limits<int>::max());
static_assert(saturating_cast<int>(-1.0e30) == std::numeric_limits<int>::min());
static_assert(saturating_cast<unsigned>(1.0e30) ==
              std::numeric_limits<unsigned>::max());

// Negative input to an unsigned target -> 0.
static_assert(saturating_cast<unsigned>(-1.0) == 0u);
static_assert(saturating_cast<unsigned>(-123.456) == 0u);
static_assert(saturating_cast<std::uint64_t>(-1.0e30) == 0);

// Infinities saturate.
static_assert(saturating_cast<int>(kInf) == std::numeric_limits<int>::max());
static_assert(saturating_cast<int>(-kInf) == std::numeric_limits<int>::min());
static_assert(saturating_cast<unsigned>(kInf) ==
              std::numeric_limits<unsigned>::max());
static_assert(saturating_cast<unsigned>(-kInf) == 0u);

// NaN -> 0.
static_assert(saturating_cast<int>(kNaN) == 0);
static_assert(saturating_cast<unsigned>(kNaN) == 0u);
static_assert(saturating_cast<std::int64_t>(kNaN) == 0);

// Small widths.
static_assert(saturating_cast<std::int8_t>(127.999) == 127);
static_assert(saturating_cast<std::int8_t>(128.0) == 127);
static_assert(saturating_cast<std::int8_t>(-128.0) == -128);
static_assert(saturating_cast<std::int8_t>(-129.0) == -128);
static_assert(saturating_cast<std::uint8_t>(255.9) == 255);
static_assert(saturating_cast<std::uint8_t>(256.0) == 255);
static_assert(saturating_cast<std::uint8_t>(-5.0) == 0);

// float and long double sources are accepted at compile time too.
static_assert(saturating_cast<short>(2.5f) == 2);
static_assert(saturating_cast<short>(2.5L) == 2);
static_assert(saturating_cast<std::int64_t>(-9.75L) == -9);

// bool as target is rejected by the constraint (design choice: bool is not
// a signed or unsigned integer type for [numeric.sat.cast] purposes), and
// non-floating sources are not this overload's business. (Checked via a
// named concept: an inline requires-expression inside static_assert is a
// hard error on GCC 13 for this shape.)
template <class R, class From>
concept can_saturating_cast = requires(From f) { saturating_cast<R>(f); };
static_assert(can_saturating_cast<int, double>);
static_assert(!can_saturating_cast<bool, double>);
static_assert(!can_saturating_cast<int, int>);

// ---------------------------------------------------------------------------
// Runtime tests.
// ---------------------------------------------------------------------------

int main() {
    // --- 64-bit signed boundaries (the case where (double)max lies) -----
    const double two63 = std::ldexp(1.0, 63);          // 2^63, exact
    assert(saturating_cast<std::int64_t>(two63) ==
           std::numeric_limits<std::int64_t>::max());
    // Largest double strictly below 2^63 is 2^63 - 1024; it must convert
    // exactly, NOT saturate.
    const double just_below_two63 = std::nextafter(two63, 0.0);
    assert(just_below_two63 == 9223372036854774784.0); // 2^63 - 1024
    assert(saturating_cast<std::int64_t>(just_below_two63) ==
           9223372036854774784LL);
    // -2^63 exactly is representable and must give min, not UB.
    assert(saturating_cast<std::int64_t>(-two63) ==
           std::numeric_limits<std::int64_t>::min());
    // Just above -2^63 converts exactly.
    const double just_above_minus_two63 = std::nextafter(-two63, 0.0);
    assert(saturating_cast<std::int64_t>(just_above_minus_two63) ==
           (-9223372036854775807LL - 1 + 1024));
    // Far outside saturates.
    assert(saturating_cast<std::int64_t>(1.0e30) ==
           std::numeric_limits<std::int64_t>::max());
    assert(saturating_cast<std::int64_t>(-1.0e30) ==
           std::numeric_limits<std::int64_t>::min());
    assert(saturating_cast<std::int64_t>(kInf) ==
           std::numeric_limits<std::int64_t>::max());
    assert(saturating_cast<std::int64_t>(-kInf) ==
           std::numeric_limits<std::int64_t>::min());
    assert(saturating_cast<std::int64_t>(kNaN) == 0);

    // --- 64-bit unsigned boundaries ---------------------------------------
    const double two64 = std::ldexp(1.0, 64);          // 2^64, exact
    assert(saturating_cast<std::uint64_t>(two64) ==
           std::numeric_limits<std::uint64_t>::max());
    // Largest double strictly below 2^64 is 2^64 - 2048.
    const double just_below_two64 = std::nextafter(two64, 0.0);
    assert(just_below_two64 == 18446744073709549568.0); // 2^64 - 2048
    assert(saturating_cast<std::uint64_t>(just_below_two64) ==
           18446744073709549568ULL);
    assert(saturating_cast<std::uint64_t>(-0.5) == 0);
    assert(saturating_cast<std::uint64_t>(-1.0e30) == 0);
    assert(saturating_cast<std::uint64_t>(-kInf) == 0);
    assert(saturating_cast<std::uint64_t>(kInf) ==
           std::numeric_limits<std::uint64_t>::max());
    assert(saturating_cast<std::uint64_t>(kNaN) == 0);

    // --- long double source, 64-bit targets --------------------------------
    // long double (x86 80-bit, 64-bit significand) represents every integer
    // up to 2^64 exactly, which lets us test 2^64 - 1024 precisely.
    const long double ld_two64_minus_1024 = std::ldexp(1.0L, 64) - 1024.0L;
    assert(saturating_cast<std::uint64_t>(ld_two64_minus_1024) ==
           18446744073709550592ULL); // 2^64 - 1024
    assert(saturating_cast<std::uint64_t>(std::ldexp(1.0L, 64)) ==
           std::numeric_limits<std::uint64_t>::max());
    assert(saturating_cast<std::int64_t>(std::ldexp(1.0L, 63)) ==
           std::numeric_limits<std::int64_t>::max());
    assert(saturating_cast<std::int64_t>(-std::ldexp(1.0L, 63)) ==
           std::numeric_limits<std::int64_t>::min());
    assert(saturating_cast<std::int64_t>(
               std::numeric_limits<long double>::quiet_NaN()) == 0);
    assert(saturating_cast<std::uint32_t>(
               std::numeric_limits<long double>::infinity()) ==
           std::numeric_limits<std::uint32_t>::max());

    // --- float source -------------------------------------------------------
    assert(saturating_cast<int>(1.0e30f) == std::numeric_limits<int>::max());
    assert(saturating_cast<int>(-1.0e30f) == std::numeric_limits<int>::min());
    assert(saturating_cast<int>(std::numeric_limits<float>::quiet_NaN()) == 0);
    assert(saturating_cast<unsigned>(std::numeric_limits<float>::infinity()) ==
           std::numeric_limits<unsigned>::max());
    assert(saturating_cast<std::int16_t>(32767.0f) == 32767);
    assert(saturating_cast<std::int16_t>(32768.0f) == 32767);
    assert(saturating_cast<std::int16_t>(-32768.0f) == -32768);
    assert(saturating_cast<std::int16_t>(-32769.0f) == -32768);

    // --- every width, spot values -------------------------------------------
    assert(saturating_cast<std::int8_t>(-0.0) == 0);
    assert(saturating_cast<std::uint8_t>(-0.0) == 0);
    assert(saturating_cast<std::int16_t>(12345.678) == 12345);
    assert(saturating_cast<std::uint16_t>(65535.9) == 65535);
    assert(saturating_cast<std::uint16_t>(65536.0) == 65535);
    assert(saturating_cast<std::int32_t>(-2147483648.0) ==
           std::numeric_limits<std::int32_t>::min());
    assert(saturating_cast<std::uint32_t>(4294967295.0) ==
           std::numeric_limits<std::uint32_t>::max());
    assert(saturating_cast<std::uint32_t>(4294967296.0) ==
           std::numeric_limits<std::uint32_t>::max());

    // --- exhaustive-ish sweep for 8-bit targets against an independent ----
    // reference that is trustworthy at this width because (double)max and
    // (double)min are exact for 8/16-bit targets.
    for (double v = -300.0; v <= 300.0; v += 0.25) {
        double t = std::trunc(v);
        int ref_s = t > 127 ? 127 : (t < -128 ? -128 : static_cast<int>(t));
        unsigned ref_u = t > 255 ? 255u : (t < 0 ? 0u : static_cast<unsigned>(t));
        assert(saturating_cast<std::int8_t>(v) == ref_s);
        assert(saturating_cast<std::uint8_t>(v) == ref_u);
    }
    for (double v = -70000.0; v <= 70000.0; v += 13.75) {
        double t = std::trunc(v);
        int ref_s = t > 32767 ? 32767 : (t < -32768 ? -32768 : static_cast<int>(t));
        assert(saturating_cast<std::int16_t>(v) == ref_s);
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
