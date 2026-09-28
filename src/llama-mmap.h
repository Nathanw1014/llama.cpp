#pragma once

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
