# saturating-cast-fp

Portable C++20 reference implementation, 64-bit boundary test suite, and microbenchmarks for extending `std::saturating_cast` to floating-point inputs (WG21 P4355).

## Files
* `saturating_cast_fp.hpp` — Header-only C++20 reference implementation (`constexpr` / `noexcept`) comparing against the exact bound `2^digits(R)` (guarded by `digits(R) < max_exponent(From)`).
* `test_saturating_cast_fp.cpp` — Compile-time (`static_assert`) and runtime boundary test suite (`2^63`, `2^63 - 1024`, `-2^63`, `2^64 - 2048` in `double`, `2^64 - 1024` in `long double`, `NaN`, `±inf`, and 8/16-bit sweeps).
* `bench.cpp` — Microbenchmark comparing the reference implementation against naive clamp-then-cast and bare `static_cast`.
* `verification-transcript-2026-10-13.txt` — Test and benchmark outputs on GCC 13.3.0 and Clang 18.1.3 (C++20 / C++23).

## Build & Run
```sh
g++ -std=c++20 -O2 -Wall -Wextra -Werror test_saturating_cast_fp.cpp -o test_sat && ./test_sat
g++ -std=c++20 -O2 bench.cpp -o bench && ./bench
