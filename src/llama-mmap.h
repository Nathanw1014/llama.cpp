#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <utility>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// The whole pages inside [first, last) that no keep range touches, as [first, last) byte ranges: what a load may drop from
// the page cache once [first, last) has been copied into a device buffer. A page the copied range shares with a neighbour
// (partial at either edge) is kept, and a keep range claims every page it overlaps, so nothing a host-mapped tensor (or a
// lazily read one) is served from at run time is dropped. page must be a power of two; keep need not be sorted.
inline std::vector<std::pair<size_t, size_t>> llama_drop_cache_ranges(size_t first, size_t last,
        const std::vector<std::pair<size_t, size_t>> & keep, size_t page) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t b = (first + page - 1) & ~(page - 1);
    const size_t e = last & ~(page - 1);
    if (last <= first || b >= e) {
        return out;
    }
    std::vector<std::pair<size_t, size_t>> k;
    for (const auto & r : keep) {
        if (r.second <= r.first) {
            continue;
        }
        const size_t kb = r.first & ~(page - 1);
        const size_t ke = (r.second + page - 1) & ~(page - 1);
        if (ke > b && kb < e) {
            k.emplace_back(kb < b ? b : kb, ke > e ? e : ke);
        }
    }
    std::sort(k.begin(), k.end());
    size_t cur = b;
    for (const auto & r : k) {
        if (r.first > cur) {
            out.emplace_back(cur, r.first);
        }
        cur = r.second > cur ? r.second : cur;
    }
    if (cur < e) {
        out.emplace_back(cur, e);
    }
    return out;
}

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    // after [first, last) has been copied out of the mapping into another buffer: drop its pages from this process's page
    // tables (madvise) and from the page cache (posix_fadvise; mapped pages are not evicted, hence the madvise first).
    // Only the pages llama_drop_cache_ranges(first, last, keep) returns. Clean, read-only pages: the data is reread from
    // the file if anything touches them again. Returns the bytes advised (0 where this is not supported).
    size_t drop_cache(size_t first, size_t last, const ranges & keep);

    // true if [ptr, ptr + len) lies inside this mapping
    bool contains(const void * ptr, size_t len) const;

    // ask the kernel to start reading the given rows. issued as one batch so the faults overlap
    // instead of serializing.
    // n_threads > 1 issues the hints from that many threads (each hint is a ~10 us syscall even on a
    // cached page); probe = hint only the pages mincore() reports absent.
    void prefetch_rows(const void * base, size_t stride, size_t row_size,
                       const int32_t * rows, size_t n_rows, int n_threads = 1, bool probe = false) const;

    // an O_DIRECT descriptor on the mapped file (opened lazily, owned by the mapping), or -1: lets a
    // gather read scattered rows with pread() at NVMe queue depth instead of one page fault each
    int direct_fd() const;

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};


struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
