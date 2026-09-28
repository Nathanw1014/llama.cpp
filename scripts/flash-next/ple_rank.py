#!/usr/bin/env python3
# Rank PLE table rows by lookup frequency over tokenized corpora (int32 token files, one sequence
# each), reproducing qwen4exp_ple_rows() exactly; the hash constants come from the GGUF.
# Output: rank.npy, int32 global row ids, most frequent first (input to make_ple_hot.py).
#   ple_rank.py --gguf model-00001.gguf --out rank.npy corpus1.i32 [corpus2.i32 ...]
# Check against the model with LLAMA_PLE_DUMP=<file> (it appends every ubatch's row ids).
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gguf-py'))
from gguf import GGUFReader

def hparams(path):
    r = GGUFReader(path)
    arch = str(bytes(r.fields['general.architecture'].parts[-1]), 'utf-8')
    def get(k):
        f = r.fields[f'{arch}.ple.{k}']
        v = [f.parts[i][0] for i in f.data]
        return np.array(v, dtype=np.uint64)
    return dict(n=int(get('ngram_size')[0]), per=int(get('heads_per_ngram')[0]), eos=int(get('eos_token_id')[0]),
                mult=get('layer_multipliers'), off=get('head_offsets'), voc=get('head_vocab_sizes'))

def rows_for_seq(hp, tok):
    """global row ids [T, n_heads] for one sequence that starts fresh (predecessors read as EOS)"""
    tok = np.asarray(tok, dtype=np.int64)
    T, n = len(tok), hp['n']
    ctx = [tok]
    cut = np.zeros(T, bool)
    for s in range(1, n):
        p = np.full(T, -1, np.int64); p[s:] = tok[:-s]
        cut = cut | (p < 0) | (p == hp['eos'])        # an EOS resets everything at or before it
        ctx.append(np.where(cut, hp['eos'], p))
    out = np.empty((T, (n - 1) * hp['per']), np.int64)
    with np.errstate(over='ignore'):
        mixed = ctx[0].astype(np.uint64) * hp['mult'][0]
        for g in range(2, n + 1):
            mixed = mixed ^ (ctx[g - 1].astype(np.uint64) * hp['mult'][g - 1])
            for j in range(hp['per']):
                h = (g - 2) * hp['per'] + j
                out[:, h] = (mixed % hp['voc'][h] + hp['off'][h]).astype(np.int64)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('corpora', nargs='+')
    a = ap.parse_args()
    hp = hparams(a.gguf)
    r = np.concatenate([rows_for_seq(hp, np.fromfile(c, dtype=np.int32)).ravel() for c in a.corpora])
    u, cnt = np.unique(r, return_counts=True)
    o = np.argsort(-cnt, kind='stable')
    np.save(a.out, u[o].astype(np.int32))
    print(f'{len(r)} lookups, {len(u)} distinct rows, top-1M covers {cnt[o][:1<<20].sum()/len(r)*100:.1f}% of the training lookups')
