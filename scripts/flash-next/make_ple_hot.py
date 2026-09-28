#!/usr/bin/env python3
# Build a PLE hot-row sidecar for LLAMA_PLE_HOT: the top-N most frequent PLE table rows of a
# tokenized corpus, copied byte for byte out of the GGUF (lossless, any row type).
#
#   make_ple_hot.py --gguf shard-with-table.gguf --rank rank.npy --n 4194304 --out ple-hot.bin
#
# rank.npy: int32 global row ids, most frequent first (see ple_rows.py / the fn-plehot record).
# File layout (little endian):
#   char[8] "PLEHOT01" | u32 row_bytes | u32 ggml_type | u64 n_table_rows | u64 n_hot
#   i32 ids[n_hot] (ascending) | zero pad to 64 | u8 rows[n_hot][row_bytes] (same order as ids)
import argparse, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gguf-py'))
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument('--gguf', required=True)
ap.add_argument('--rank', required=True)
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tensor', default='per_layer_token_embd.weight')
a = ap.parse_args()

r = GGUFReader(a.gguf)
t = next(t for t in r.tensors if t.name == a.tensor)
n_rows = int(t.shape[1])
row_bytes = t.n_bytes // n_rows
assert row_bytes * n_rows == t.n_bytes
base = int(t.data_offset)
ttype = int(t.tensor_type)
del r

ids = np.load(a.rank)[:a.n].astype(np.int64)
assert ids.min() >= 0 and ids.max() < n_rows
ids = np.unique(ids)
n = len(ids)
print(f'{a.tensor}: type {t.tensor_type.name} rows {n_rows} x {row_bytes} B at {base}; hot {n} rows = {n*row_bytes/2**20:.1f} MiB')

out = np.empty((n, row_bytes), np.uint8)
CH = 256 << 20  # sequential chunks: one pass over the table instead of millions of page faults
fd = os.open(a.gguf, os.O_RDONLY)
j = 0
for c0 in range(0, n_rows * row_bytes, CH):
    c1 = min(n_rows * row_bytes, c0 + CH)
    r0, r1 = c0 // row_bytes, (c1 + row_bytes - 1) // row_bytes
    k1 = np.searchsorted(ids, r1)
    if k1 > j:
        lo = ids[j] * row_bytes
        hi = (ids[k1 - 1] + 1) * row_bytes
        buf = np.frombuffer(os.pread(fd, hi - lo, base + lo), np.uint8)
        offs = (ids[j:k1] * row_bytes - lo)[:, None] + np.arange(row_bytes)[None, :]
        out[j:k1] = buf[offs]
        j = k1
    os.posix_fadvise(fd, base + c0, c1 - c0, os.POSIX_FADV_DONTNEED)  # leave the page cache as found
os.close(fd)
assert j == n

with open(a.out, 'wb') as f:
    f.write(b'PLEHOT01' + struct.pack('<IIQQ', row_bytes, ttype, n_rows, n))
    f.write(ids.astype('<i4').tobytes())
    f.write(b'\0' * ((-f.tell()) % 64))
    f.write(out.tobytes())
print('wrote', a.out, os.path.getsize(a.out))
