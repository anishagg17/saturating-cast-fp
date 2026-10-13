// bench.cpp
//
// Micro-benchmark addressing the concern raised in the original
// std-proposals thread: that a generic, portable pure-C++ implementation
// of a saturating FP -> integer cast is suboptimal on many ISAs compared
// with a compiler builtin.
//
// We compare, on identical input data:
//   * std_proposal::saturating_cast (this proposal's reference
//     implementation -- branchy, correct at the 2^63 / 2^64 boundary)
//   * a "naive" saturating cast as commonly hand-written: compare against
//     (double)numeric_limits<R>::max() / min() and then static_cast.
//     NOTE: the naive version is subtly WRONG for 64-bit targets --
//     (double)INT64_MAX rounds up to 2^63 and (double)UINT64_MAX to 2^64,
//     so at the boundary it can still reach the UB static_cast. It is
//     included only as the performance baseline users write today; the
//     honest question is what the *correct* boundary costs.
//   * a bare static_cast (undefined behaviour out of range -- shown only
//     as a lower bound on cost; all inputs here are in range for it).
//
// Methodology: one large array of doubles spanning in-range, boundary and
// out-of-range values, many passes, results accumulated into a checksum
// that is printed so nothing is optimised away. Numbers are single-run
// wall times on whatever machine this was built on -- indicative, not a
// substitute for the per-ISA study the paper calls for.

#include "saturating_cast_fp.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

template <class R>
inline R naive_saturating_cast(double x) noexcept {
    if (x != x) return R{0}; // NaN
    if (x > static_cast<double>(std::numeric_limits<R>::max()))
        return std::numeric_limits<R>::max();
    if (x < static_cast<double>(std::numeric_limits<R>::min()))
        return std::numeric_limits<R>::min();
    return static_cast<R>(x); // reachable with an unrepresentable value
                              // for 64-bit R near the boundary (see above)
}

template <class F>
double time_it(const char* name, const std::vector<double>& data,
               int passes, F&& f) {
    // One warm-up pass, then timed passes; checksum escaped via printf.
    std::uint64_t sink = f(data);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < passes; ++i)
        sink += f(data);
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per =
        secs * 1e9 / (static_cast<double>(passes) * data.size());
    std::printf("%-44s %8.3f ns/elem   (checksum %llu)\n", name, ns_per,
                static_cast<unsigned long long>(sink));
    return ns_per;
}

template <class R>
std::uint64_t run_proposed(const std::vector<double>& d) {
    std::uint64_t s = 0;
    for (double x : d)
        s += static_cast<std::uint64_t>(std_proposal::saturating_cast<R>(x));
    return s;
}
template <class R>
std::uint64_t run_naive(const std::vector<double>& d) {
    std::uint64_t s = 0;
    for (double x : d)
        s += static_cast<std::uint64_t>(naive_saturating_cast<R>(x));
    return s;
}
std::uint64_t run_bare_i64(const std::vector<double>& d) {
    std::uint64_t s = 0;
    for (double x : d) // caller guarantees in-range, non-NaN inputs
        s += static_cast<std::uint64_t>(static_cast<std::int64_t>(x));
    return s;
}

} // namespace

int main() {
    constexpr std::size_t kN = 1u << 16;
    std::vector<double> mixed;
    mixed.reserve(kN);
    // Deterministic pseudo-random spread over a wide range, plus specials.
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (std::size_t i = 0; i < kN; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        // roughly [-1e19, 1e19]: covers in-range and saturating cases
        const double v = static_cast<double>(state % 2000000) - 1000000.0;
        mixed.push_back(v * 1.0e13);
    }
    mixed[0] = std::numeric_limits<double>::quiet_NaN();
    mixed[1] = std::numeric_limits<double>::infinity();
    mixed[2] = -std::numeric_limits<double>::infinity();
    mixed[3] = std::ldexp(1.0, 63);   // 2^63 boundary
    mixed[4] = std::ldexp(1.0, 64);   // 2^64 boundary
    mixed[5] = -std::ldexp(1.0, 63);  // -2^63 boundary

    constexpr int kPasses = 2000;

    std::printf("=== double -> int64_t ===\n");
    time_it("proposed saturating_cast<int64_t>", mixed, kPasses,
            [](const std::vector<double>& d) { return run_proposed<std::int64_t>(d); });
    time_it("naive clamp-then-static_cast<int64_t>", mixed, kPasses,
            [](const std::vector<double>& d) { return run_naive<std::int64_t>(d); });

    std::printf("=== double -> uint64_t ===\n");
    time_it("proposed saturating_cast<uint64_t>", mixed, kPasses,
            [](const std::vector<double>& d) { return run_proposed<std::uint64_t>(d); });
    time_it("naive clamp-then-static_cast<uint64_t>", mixed, kPasses,
            [](const std::vector<double>& d) { return run_naive<std::uint64_t>(d); });

    // Bare-cast lower bound on an all-in-range data set.
    std::vector<double> in_range;
    in_range.reserve(kN);
    for (std::size_t i = 0; i < kN; ++i)
        in_range.push_back(static_cast<double>(static_cast<int>(i) - 32768) * 1.5);
    std::printf("=== double -> int64_t, in-range only (cost floor) ===\n");
    time_it("proposed saturating_cast<int64_t>", in_range, kPasses,
            [](const std::vector<double>& d) { return run_proposed<std::int64_t>(d); });
    time_it("bare static_cast<int64_t> (UB out of range)", in_range, kPasses,
            [](const std::vector<double>& d) { return run_bare_i64(d); });

    return 0;
}
