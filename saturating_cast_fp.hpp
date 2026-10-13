// saturating_cast_fp.hpp
//
// Reference implementation for the WG21 proposal
//   "Extending std::saturating_cast to floating-point inputs"
// (extending C++26 std::saturating_cast, <numeric>, [numeric.sat.cast],
// which today is constrained to integer -> integer conversions).
//
// Implemented in namespace std_proposal (NOT std) so it can be built and
// tested against today's compilers (C++20/23; concepts + constexpr).
//
// Proposed semantics implemented here (the paper's recommended design),
// for floating-point source x and integer target R:
//   * NaN                      -> 0
//   * x too large (incl. +inf)  -> numeric_limits<R>::max()
//   * x too small (incl. -inf)  -> numeric_limits<R>::min()
//                                  (hence a negative input to an unsigned
//                                   target yields 0)
//   * otherwise                 -> truncation toward zero, exactly as a
//                                  (well-defined) static_cast<R>(x) would do
// The function is constexpr and noexcept.
//
// ---------------------------------------------------------------------------
// Why the boundary test does NOT compare against (From)max / (From)min:
//
// numeric_limits<uint64_t>::max() is 2^64 - 1, which is NOT representable
// in a double (53-bit significand); (double)max rounds UP to 2^64. A test
// like `x > (double)max` is therefore wrong at the top of the range, and
// a test like `x <= (double)max` would let 2^64 through to a static_cast
// whose truncated value is unrepresentable -- undefined behaviour. The
// same problem exists for int64_t (max = 2^63 - 1 rounds up to 2^63).
//
// Correct technique used here: compare against the exact power of two
//   B = 2^digits(R)
// Exception (added after review): when digits(R) >= max_exponent(From)
// (e.g. From = std::float16_t into a 32-bit target, or float into
// __uint128_t), B is not representable in From at all -- computing it
// overflows (to inf, or fails in constant expressions on stricter
// implementations). In that regime every finite From value already
// truncates into R's range, so only +-inf (and negative inputs for
// unsigned R) can saturate, and the code branches accordingly.
// (digits excludes the sign bit for signed types, so for int64_t B = 2^63
// and for uint64_t B = 2^64). Powers of two ARE exactly representable in
// every binary floating-point type. The truncating conversion is in range
// precisely on the half-open interval
//   signed:    -B <= x < B        (trunc(x) then lies in [min, max])
//   unsigned:   0 <= x < B
// Outside it we return the precomputed limit, so the static_cast in the
// final line is only ever executed when its result is representable, i.e.
// it is never UB. Note the signed low side: for -B-1 < x < -B, trunc(x)
// equals min, and the clamp `x < -B -> min` returns that same value, so a
// single comparison handles both cases.
// B itself is computed with a constexpr multiplication loop because
// std::ldexp is not constexpr in C++20/23, and 2^64 does not fit in any
// integer type to be converted from.
// ---------------------------------------------------------------------------

#ifndef STD_PROPOSAL_SATURATING_CAST_FP_HPP
#define STD_PROPOSAL_SATURATING_CAST_FP_HPP

#include <concepts>
#include <limits>
#include <type_traits>

namespace std_proposal {

namespace detail_sat_fp {

// 2^digits(R) as a value of floating-point type From.
// A loop is used (at most 64 iterations, all at compile time in constant
// expressions) because no integer type can hold 2^64 for conversion.
template <class R, std::floating_point From>
constexpr From two_pow_digits() noexcept {
    From p = From(1);
    for (int i = 0; i < std::numeric_limits<R>::digits; ++i)
        p *= From(2);
    return p;
}

} // namespace detail_sat_fp

// Target-type constraint: a signed or unsigned integer type, mirroring
// [numeric.sat.cast]. std::integral alone is too wide -- it also admits
// bool and the character types (char, wchar_t, char8_t, char16_t,
// char32_t), none of which are integer types in the standard's taxonomy
// (signed char / unsigned char ARE admitted). Extended integer types
// (e.g. __int128) remain admissible where the implementation has them --
// hence the numeric_limits arm of the concept: std::is_integral is
// false for __int128 in libstdc++, although __int128 is an extended
// integer type in the standard's taxonomy and numeric_limits knows it.
// For the same reason the implementation tests numeric_limits<R>::is_signed
// rather than std::is_signed_v<R> (also false for __int128 in libstdc++,
// which once sent negative inputs down the unsigned branch).
template <class T>
concept std_integer =
    (std::integral<T> ||
     (std::numeric_limits<std::remove_cv_t<T>>::is_specialized &&
      std::numeric_limits<std::remove_cv_t<T>>::is_integer)) &&
    !std::same_as<std::remove_cv_t<T>, bool> &&
    !std::same_as<std::remove_cv_t<T>, char> &&
    !std::same_as<std::remove_cv_t<T>, wchar_t> &&
    !std::same_as<std::remove_cv_t<T>, char8_t> &&
    !std::same_as<std::remove_cv_t<T>, char16_t> &&
    !std::same_as<std::remove_cv_t<T>, char32_t>;

// Floating-point -> integer overload (the proposed addition).
// R: a signed or unsigned integer type (see std_integer above).
// From: a floating-point type.
template <class R, std::floating_point From>
    requires std_integer<R>
constexpr R saturating_cast(From x) noexcept {
    // NaN -> 0. Written as x != x (rather than std::isnan) so the function
    // stays usable in constant expressions with C++20/23 libraries, where
    // std::isnan is not constexpr. (Under -ffast-math / -ffinite-math-only
    // this test may be optimised away; there NaN cannot occur by the
    // program's own contract -- a QoI point the paper discusses.)
    if (x != x)
        return R{0};

    if constexpr (std::numeric_limits<R>::digits <
                  std::numeric_limits<From>::max_exponent) {
        constexpr From bound =
            detail_sat_fp::two_pow_digits<R, From>(); // 2^digits

        if constexpr (std::numeric_limits<R>::is_signed) {
            if (x >= bound) // includes +inf
                return std::numeric_limits<R>::max();
            if (x < -bound) // includes -inf; see header comment for the
                            // interval (-bound-1, -bound), which also maps
                            // to min under truncation
                return std::numeric_limits<R>::min();
        } else {
            if (x < From(0)) // any negative value, including -inf -> 0
                return std::numeric_limits<R>::min(); // == 0 for unsigned
            if (x >= bound)  // includes +inf
                return std::numeric_limits<R>::max();
        }

        // Here the value truncated toward zero is representable in R
        // (signed: -bound <= x < bound; unsigned: 0 <= x < bound),
        // so this conversion is well-defined and truncates toward zero.
        return static_cast<R>(x);
    } else {
        // digits(R) >= max_exponent(From): 2^digits is not representable
        // in From, but every finite From value satisfies |x| < 2^digits,
        // so its truncation is always representable in R. Only the
        // infinities (and negative inputs for unsigned R) saturate.
        constexpr From inf = std::numeric_limits<From>::infinity();
        if constexpr (std::numeric_limits<R>::is_signed) {
            if (x == inf)
                return std::numeric_limits<R>::max();
            if (x == -inf)
                return std::numeric_limits<R>::min();
        } else {
            if (x < From(0)) // any negative value, including -inf -> 0
                return std::numeric_limits<R>::min(); // == 0 for unsigned
            if (x == inf)
                return std::numeric_limits<R>::max();
        }
        return static_cast<R>(x);
    }
}

// Integer -> integer behaviour is NOT re-implemented here: it already
// exists as C++26 std::saturating_cast. Nothing in this header modifies
// namespace std.

} // namespace std_proposal

#endif // STD_PROPOSAL_SATURATING_CAST_FP_HPP
