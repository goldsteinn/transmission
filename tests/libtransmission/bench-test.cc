#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#include <type_traits>
#include <iostream>
#include <vector>
#include <iterator>

#include "transmission.h"
#include "crypto-utils.h"
#include "bitfield.h"


#define ALWAYS_INLINE inline __attribute__((always_inline))
#define NEVER_INLINE  __attribute__((noinline))
#define CONST_ATTR    __attribute__((const))
#define PURE_ATTR     __attribute__((pure))
#define BENCH_FUNC    __attribute__((noinline, noclone, aligned(4096)))

#define COMPILER_OOE_BARRIER() asm volatile("lfence" : : : "memory")
#define OOE_BARRIER()          asm volatile("lfence" : : :)
#define COMPILER_BARRIER()     asm volatile("" : : : "memory");
#define COMPILER_DO_NOT_OPTIMIZE_OUT(X)                                        \
    asm volatile("" : : "i,r,m"(X) : "memory")
#define COMPILER_DO_NOT_OPTIMIZE_OUT_FENCE(X)                                  \
    asm volatile("lfence" : : "i,r,m"(X) : "memory")

#define I__CAT(X, Y)   X##Y
#define CAT(X, Y)      I__CAT(X, Y)
#define I__V_TO_STR(X) #X
#define V_TO_STR(X)    I__V_TO_STR(X)

#define NO_LSD_RD(tmp, r) "pop " #tmp "\nmovq " #r ", %%rsp\n"
#define NO_LSD_WR(tmp, r) "push " #tmp "\nmovq " #r ", %%rsp\n"

#define IMPOSSIBLE(X)                                                          \
    if (X) {                                                                   \
        __builtin_unreachable();                                               \
    }

#define PRINT(...) fprintf(stderr, __VA_ARGS__)

struct timer {
    static constexpr clockid_t cid       = CLOCK_MONOTONIC;
    static constexpr uint64_t  sec_to_ns = 1000 * 1000 * 1000;

    using time_t = struct timespec;

    time_t tstart;
    time_t tend;

    const char * const
    units() const {
        return "ns";
    }

    void ALWAYS_INLINE
    start() {
        clock_gettime(cid, &tstart);
        COMPILER_OOE_BARRIER();
    }

    void ALWAYS_INLINE
    end() {
        COMPILER_OOE_BARRIER();
        clock_gettime(cid, &tend);
    }

    uint64_t
    get_start() const {
        return sec_to_ns * tstart.tv_sec + tstart.tv_nsec;
    }
    uint64_t
    get_end() const {
        return sec_to_ns * tend.tv_sec + tend.tv_nsec;
    }

    uint64_t
    dif() {
        return get_end() - get_start();
    }

    double
    ddif() {
        return ((double)dif());
    }

    double
    ddif(uint64_t n) {
        return ddif() / ((double)n);
    }

    void
    std_print() {
        std_print("", 1);
    }

    void
    std_print(const char * const hdr) {
        std_print(hdr, 1);
    }

    void
    std_print(uint64_t n) {
        std_print("", n);
    }

    void
    std_print(const char * const hdr, uint64_t n) {
        if (hdr == NULL || hdr[0] == 0) {
            fprintf(stderr, "%.3E %s\n", ddif(n), units());
        }
        else {
            fprintf(stderr, "%-8s: %.3E %s\n", hdr, ddif(n), units());
        }
    }
};

enum { NRAND_CONFS = 16384, NRAND_TRIALS = 2048, NFIXED_TRIALS = 1048576 };

#define make_rand_bench(type, sink)                                            \
    double BENCH_FUNC CAT(bench_rand_, type)(tr_bitfield const & bf,         \
                                               uint32_t const *    bounds_) {      \
        timer t;                                                               \
                                                                               \
        t.start();                                                             \
        for (uint32_t trials = NRAND_TRIALS; trials; --trials) {               \
            uint32_t const *    bounds = bounds_;                       \
            for (uint32_t const * end = bounds + NRAND_CONFS; bounds != end;   \
                 ++bounds) {                                                   \
                uint32_t bound = *bounds;                                      \
                                                                               \
                uint32_t lb = bound & 0xffff;                                  \
                uint32_t ub = bound >> 16;                                     \
                sink(bf.count(lb, ub));                                        \
            }                                                                  \
        }                                                                      \
                                                                               \
        t.end();                                                               \
                                                                               \
        return t.ddif(NRAND_TRIALS * NRAND_CONFS);                             \
    }


#define make_fixed_bench(type, sink)                                           \
    double BENCH_FUNC CAT(bench_fixed_, type)(tr_bitfield const & bf,        \
                                                uint32_t lb, uint32_t ub) {    \
        timer t;                                                               \
                                                                               \
        t.start();                                                             \
        for (uint32_t trials = NFIXED_TRIALS; trials; --trials) {              \
            sink(bf.count(lb, ub));                                            \
        }                                                                      \
                                                                               \
        t.end();                                                               \
                                                                               \
        return t.ddif(NRAND_TRIALS * NRAND_CONFS);                             \
    }


make_rand_bench(lat, COMPILER_DO_NOT_OPTIMIZE_OUT_FENCE);
make_rand_bench(tput, COMPILER_DO_NOT_OPTIMIZE_OUT);

make_fixed_bench(lat, COMPILER_DO_NOT_OPTIMIZE_OUT_FENCE);
make_fixed_bench(tput, COMPILER_DO_NOT_OPTIMIZE_OUT);

enum { FIXED = 0, RAND = 1, LAT = 2, TPUT = 3 };

typedef struct result {
    uint32_t todo;
    uint32_t type;
    uint32_t lb;
    uint32_t ub;

    double time;
} result_t;


result_t *
run_rand_benchmark(result_t * results, tr_bitfield const & bf) {
    uint32_t * bounds = (uint32_t *)calloc(NRAND_CONFS, sizeof(uint32_t));

    for (uint32_t range = 8; range <= 8192; range += range) {
        for (uint32_t i = 0; i < NRAND_CONFS; ++i) {
            uint32_t sz    = rand() % range;
            /* Need non-zero value. */
            while(sz == 0) {
                sz    = rand() % range;
            }
            uint32_t slack = 8192 - sz;


            uint32_t lb = slack ? (rand() % slack) : 0;

            uint32_t bound = lb | ((lb + sz) << 16);

            bounds[i] = bound;
        }
        results->todo = RAND;
        results->type = TPUT;
        results->lb   = 0;
        results->ub   = range;

        results->time = bench_rand_tput(bf, bounds);

        ++results;

        results->todo = RAND;
        results->type = LAT;
        results->lb   = 0;
        results->ub   = range;

        results->time = bench_rand_lat(bf, bounds);

        ++results;
    }
    free(bounds);
    return results;
}

result_t *
run_fixed_benchmark(result_t * results, tr_bitfield const & bf) {

    /* Changing lb beyond 0/1 doesn't really do much for perf. */
    for (uint32_t lb = 0; lb < 2; ++lb) {
        uint32_t ub = lb + 1;
        while (1) {
            results->type = FIXED | TPUT;
            results->lb   = lb;
            results->ub   = ub;

            results->time = bench_fixed_tput(bf, lb, ub);

            ++results;

            results->type = FIXED | LAT;
            results->lb   = lb;
            results->ub   = ub;

            results->time = bench_fixed_lat(bf, lb, ub);

            if (ub < 64) {
                ub += 7;
            }
            else if (ub < (8192 - 256)) {
                ub += 256;
            }
            else if (ub == 8192) {
                break;
            }
            else {
                ub = 8192;
            }
        }
    }
    return results;
}

void
print_res(result_t const * res) {
    const char * todo = res->todo == FIXED ? "FIXED" : "RAND";
    const char * type = res->type == LAT ? "LAT" : "TPUT";

    printf("%-8s,%-8s,%-8u,%-8u,%10.3lf\n", todo, type, res->lb,
            res->ub, res->time);
}

int
main() {
    result_t * res     = (result_t *)calloc(16384, sizeof(result_t));
    result_t * res_end = res;
    tr_bitfield bf(8192);
    for(uint32_t i =0; i < 8192; i += 17) {
        bf.set(i, 1);
    }

    res_end = run_fixed_benchmark(res_end, bf);
    res_end = run_rand_benchmark(res_end, bf);

    for (; res != res_end; ++res) {
        print_res(res);
    }
}
