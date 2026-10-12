// llama_drop_cache_ranges (LLAMA_LOAD_DROP_CACHE): the page ranges a mmap load may drop from the page cache after it has
// copied a tensor's file range into a device buffer. The loader advises only the COPY branch of load_all_data, and only
// after ggml_backend_tensor_set has returned: with mmap there is no async upload backend (the pinned staging buffers and
// their events serve only the non-mmap read path), so no advised range is still being read when it is dropped. What this
// test pins down is the range arithmetic: only whole pages strictly inside the copied range, never a page shared with a
// neighbour, and never a page a keep range (a host-mapped tensor such as the PLE n-gram table, or a lazily read range)
// touches. CPU only, no model.

#include "../src/llama-mmap.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using ranges = std::vector<std::pair<size_t, size_t>>;

static int n_fail = 0;

static void expect(const char * what, const ranges & got, const ranges & want) {
    if (got != want) {
        ++n_fail;
        fprintf(stderr, "FAIL %s: got", what);
        for (const auto & r : got) fprintf(stderr, " [%zu, %zu)", r.first, r.second);
        fprintf(stderr, ", want");
        for (const auto & r : want) fprintf(stderr, " [%zu, %zu)", r.first, r.second);
        fprintf(stderr, "\n");
    }
}

int main() {
    const size_t P = 4096;

    // edges: the partial first and last pages are shared with the neighbouring tensors and stay cached
    expect("interior pages", llama_drop_cache_ranges(100, 3*P + 50, {}, P), {{P, 3*P}});
    expect("aligned tensor", llama_drop_cache_ranges(2*P, 5*P, {}, P), {{2*P, 5*P}});
    expect("within one page", llama_drop_cache_ranges(10, P - 10, {}, P), {});
    expect("spanning one boundary", llama_drop_cache_ranges(P - 10, P + 10, {}, P), {});
    expect("empty", llama_drop_cache_ranges(3*P, 3*P, {}, P), {});

    // a host-mapped neighbour claims every page it touches, even partially
    expect("host-mapped neighbour after", llama_drop_cache_ranges(P, 5*P, {{4*P + 1, 9*P}}, P), {{P, 4*P}});
    expect("host-mapped neighbour before", llama_drop_cache_ranges(0, 5*P, {{0, P + 1}}, P), {{2*P, 5*P}});
    // a lazily read range inside (as an alias would be), unsorted and overlapping keep ranges
    expect("keep inside", llama_drop_cache_ranges(0, 10*P, {{6*P + 7, 7*P}, {2*P + 5, 3*P}, {2*P, 2*P + 9}}, P),
           {{0, 2*P}, {3*P, 6*P}, {7*P, 10*P}});
    // a TENSOR_DUPLICATED range kept as a whole by its host-mapped twin: nothing to drop
    expect("same range kept", llama_drop_cache_ranges(3*P + 9, 8*P + 9, {{3*P + 9, 8*P + 9}}, P), {});
    // keep ranges elsewhere in the file do not matter
    expect("keep elsewhere", llama_drop_cache_ranges(4*P, 6*P, {{0, 4*P}, {6*P, 9*P}, {50*P, 60*P}}, P), {{4*P, 6*P}});
    // an empty keep range claims nothing
    expect("empty keep", llama_drop_cache_ranges(4*P, 6*P, {{5*P, 5*P}}, P), {{4*P, 6*P}});

    // property check against a page-by-page reference: a page is dropped iff it lies wholly inside [first, last) and no
    // keep range overlaps it; the output is sorted, disjoint, page-aligned and never adjacent (maximal runs)
    std::mt19937_64 rng(42);
    for (int it = 0; it < 20000; ++it) {
        const size_t file = 64 * P;
        size_t first = rng() % file, last = rng() % file;
        if (first > last) std::swap(first, last);
        ranges keep;
        const int nk = (int) (rng() % 5);
        for (int k = 0; k < nk; ++k) {
            size_t a = rng() % file, b = rng() % file;
            if (a > b) std::swap(a, b);
            keep.emplace_back(a, b);
        }
        const ranges got = llama_drop_cache_ranges(first, last, keep, P);
        std::vector<bool> dropped(file / P, false);
        bool ok = true;
        size_t prev_end = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const auto & r = got[i];
            ok = ok && r.first % P == 0 && r.second % P == 0 && r.first < r.second && (i == 0 || r.first > prev_end);
            prev_end = r.second;
            for (size_t p = r.first / P; p < r.second / P && p < dropped.size(); ++p) dropped[p] = true;
        }
        for (size_t p = 0; p < dropped.size(); ++p) {
            bool want = p * P >= first && (p + 1) * P <= last;
            for (const auto & k : keep) {
                if (k.first < k.second && k.first < (p + 1) * P && k.second > p * P) want = false;
            }
            ok = ok && dropped[p] == want;
        }
        if (!ok) {
            ++n_fail;
            fprintf(stderr, "FAIL property: first %zu last %zu keeps %d\n", first, last, nk);
            if (n_fail > 5) break;
        }
    }

    if (n_fail) {
        fprintf(stderr, "%d failures\n", n_fail);
        return 1;
    }
    printf("test-load-drop-cache: OK\n");
    return 0;
}
