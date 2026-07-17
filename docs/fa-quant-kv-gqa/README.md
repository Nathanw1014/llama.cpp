# Quantized-KV flash-attention decode does `gqa_ratio`× redundant dequantization (CUDA/HIP `fattn-vec`)

## TL;DR

At decode with quantized KV, `ggml_cuda_get_best_fattn_kernel` selects `BEST_FATTN_KERNEL_VEC`. The vec
kernel assigns **one Q head per block** and fuses dequantization into the KQ dot product, so the
dequantized KV value is never materialized and cannot be shared. With `gqa_ratio = 8`, the same KV head is
streamed and dequantized by **8 separate blocks**. At decode there is nothing to amortize that against
(`n = 1`), so quantized-KV decode does `gqa_ratio`× more dequant work than necessary.

Teaching the `tile` kernel to dequantize KV on load into its existing SRAM tile — it already batches
`ncols2` Q heads per block — removes the redundancy. On a Radeon 8060S (gfx1151), Qwen3-Coder-30B-A3B
(head_dim 128, gqa_ratio 8), tg32 t/s:

| depth | q8_0 today | q8_0 patched | q4_0 today | q4_0 patched | f16 |
| ----- | ---------- | ------------ | ---------- | ------------ | ----- |
| 16k   | 26.59      | **43.20**    | 26.29      | **45.69**    | 41.20 |
| 32k   | 16.65      | **38.14**    | 17.55      | **39.97**    | 31.97 |
| 64k   | 8.86       | **28.96**    | 9.35       | **31.65**    | 22.17 |

Full decode curve, **all arms same session, same commit, same box** (only the backend and the patch
differ) — q8_0 KV, tg32 t/s:

| depth | ROCm stock | ROCm + fix | RADV (vanilla Vulkan) |
| ----- | ---------- | ---------- | --------------------- |
| 0     | 53.89      | 55.14      | 65.45                 |
| 4k    | 42.25      | 51.45      | 56.77                 |
| 16k   | 26.67      | **44.59**  | 47.55                 |
| 32k   | 16.74      | **37.97**  | 38.25                 |
| 64k   | 8.95       | **29.69**  | 27.83                 |
| 128k  | 4.68       | **19.77**  | 17.85                 |

ROCm goes from **3.1x behind RADV at 64k to 6.7% ahead**, reaches parity at 32k, and is +10.8% at 128k
(where the fix is worth **+322%**). RADV is the natural control here: its FA path batches GQA and never had
this defect, so it is what the CUDA/HIP path should have been matching all along.

Quantized KV decode goes from well behind f16 to ahead of it. f16 is unchanged; prefill is unaffected.
Correctness: 6000/6000 `test-backend-ops -o FLASH_ATTN_EXT` against the CPU reference, and greedy
generation is byte-identical to the stock path.

I have a working patch series, but the dispatch condition is a design call I'd rather agree on first —
questions at the bottom.

---

## The mechanism

`fattn-vec.cuh`:

```c
const int head      = blockIdx.z - sequence*ne02;   // one Q head per block
const int gqa_ratio = ne02 / ne12;
K += nb13*sequence + nb12*(head / gqa_ratio);
V += nb23*sequence + nb22*(head / gqa_ratio);
```

32 Q heads over 4 KV heads ⇒ 8 blocks each independently stream and dequantize the same KV head.
`fattn-tile.cuh` by contrast offsets by `head0 / gqa_ratio` and batches `ncols2 = 8`, reading KV once.

So on this hardware **quantized decode and f16 decode are not the same kernel**: for `Q->ne[1] == 1`,
quantized takes `vec` while f16 falls through to `tile` (because `gqa_opt_applies`). Comparing q8_0 vs f16
decode is therefore vec-vs-tile, which matters for interpreting any measurement of this (see below).

**Note the existing guard.** In the Turing branch, f16 is explicitly steered away from `vec` under exactly
these conditions:

```c
if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
    return BEST_FATTN_KERNEL_VEC;
}
```

The quantized branch immediately below has no equivalent — I assume because `mma`/`tile` are f16-only and
`vec` is the only quantized-capable kernel, so there was nowhere else to send it. That is the gap this
addresses.

## Why "just add `ncols2` to vec" doesn't work

The obvious fix is to batch GQA heads in `vec`. It isn't sufficient. In the inner loop:

```c
for (int j = 0; j < ncols; ++j) {
    float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
```

`vec_dot_KQ` takes the **same K pointer for every column** and fuses dequant into the dot product — the
existing `ncols = 2` path already dequantizes K twice. Mapping `gqa_ratio` heads onto columns would just
move the redundancy from across-blocks to within-block.

**The fusion is what forfeits the reuse.** That costs nothing at `gqa_ratio = 1` and costs `gqa_ratio`× at
high GQA. A real fix has to split dequant from dot and materialize the dequantized K once — which is what
`launch_fattn`'s `need_f16_K/V` already does for `mma` at prefill, and what the Vulkan backend does at
decode.

## Evidence that this is dequant work, not bandwidth

The natural objection is "it's just KV bandwidth". Two controls say otherwise.

**1. q4_0 vs q8_0 on the stock path** — same kernel, same element count, 0.529× the bytes:

| depth | q8_0  | q4_0  | speedup |
| ----- | ----- | ----- | ------- |
| 16k   | 26.68 | 27.00 | 1.012×  |
| 32k   | 16.69 | 17.96 | 1.076×  |
| 64k   | 9.03  | 9.39  | 1.040×  |

Halving the KV bytes buys 1–8%. Decode time scales with KV **elements**, not bytes.

**2. The Vulkan backend, same box, same model, same KV format.** `ggml-vulkan.cpp` does:

```c
// grouped query attention - make the N dimension equal to gqa_ratio, reduce
// workgroups proportionally in y dimension.
gqa_ratio = qk_ratio;  N = gqa_ratio;  workgroups_y /= gqa_ratio;
```

It batches all 8 Q heads per workgroup, does 1/8 the dequant work, and q8_0 consequently **gains** over f16
at depth (+10…+17% at 16k–32k) instead of losing to it.

**3. Prefill is fine, and the reason confirms the shape of the defect.** q8_0 pp512 is within 1.6% of f16 at
every depth, because prefill takes `mma` via `need_f16_K/V`, which dequantizes once and amortizes over 512
query columns. One dequant costs ~1%; doing it 8× per token with zero reuse collapses decode.

## Measuring this: q8_0-vs-f16 understates it

Worth flagging for anyone reproducing. Judging the defect by the quantized-vs-f16 gap can hide it entirely,
because those arms run different kernels and the errors can cancel. On Llama-3.2-1B (head_dim 64,
gqa_ratio 4) stock q8_0 decode matches f16 within 1.5% — reads as "unaffected". It is not: the fix gives
**+22.3%** at 32k, after which q8_0 beats f16 by 19.5%. The right instrument is the same kernel with and
without the redundancy.

Scope: the redundancy applies wherever quantized decode reaches `vec` — head sizes **64, 128 and 256**
(`Q->ne[0] % 64 == 0`, `<= 256`, `!= 192`) with `gqa_ratio >= 2`. Severity is a product of `gqa_ratio` and
how much of the token budget the KV work occupies, so it ranges from ~0 (small dense model, low GQA, shallow)
to catastrophic (MoE, gqa 8, deep context). Every shape I tested gains:

| model                                    | depth | vec   | tile   | gain      |
| ---------------------------------------- | ----- | ----- | ------ | --------- |
| Llama-3.2-1B (hd 64, gqa 4, dense)       | 32k   | 83.42 | 102.06 | **+22%**  |
| Qwopus3.6-35B-A3B (hd 256, gqa 8, MoE)   | 32k   | 37.55 | 40.89  | +9%       |
| Qwen3-Coder-30B-A3B (hd 128, gqa 8, MoE) | 32k   | 16.65 | 38.14  | **+129%** |

## The patch

Two commits:

1. **`tests: cover quantized KV at head sizes 128 and 256, and gqa_ratio 8`** — stands alone, see below.
2. **`CUDA: dequantize KV on load in the tile FA kernel, use it for quantized decode`**
   - adds `flash_attn_tile_load_tile_q`, which decodes into the SRAM tile on load via the existing
     `dequantize_V_*<T, ne>` primitives, keeping the per-thread chunk (8 elements) equal across KV types
     and issuing several calls per chunk for narrower ones (`dequantize_V_q4_0` only supports `ne ∈ {2,4}`
     because it decodes nibbles through a 4-byte int; q8_0 takes 8 in one call);
   - templates `type_K`/`type_V` through `iter_KQ` → `iter` → kernel → launchers →
     `ggml_cuda_flash_attn_ext_tile_case`, **defaulted to `GGML_TYPE_F16`** so every existing call site and
     instantiation is untouched;
   - derives `need_f16_K/V` from those types instead of hardcoding `true, true` — this is the load-bearing
     line; it stops `launch_fattn` pre-dequantizing the whole cache into an f16 scratch buffer;
   - instantiates q4_0/q8_0 at head dims 64/128/256 (6 new tile instantiations);
   - prefers `tile` for symmetric quantized KV when `gqa_opt_applies`, mirroring the f16 branch.

Verification: 6000/6000 `test-backend-ops -o FLASH_ATTN_EXT`; greedy generation byte-identical to stock;
f16 arms match stock exactly (41.20 / 31.97 / 22.17); prefill neutral on every shape tested.

## Separately: quantized KV at head size 128 has no correctness coverage

This did **not** hide the defect — it's a performance bug and no correctness suite would catch it — but it
surfaced while validating the fix and is worth closing on its own.

```c
// tests/test-backend-ops.cpp
if (type_KV != GGML_TYPE_F16 && hsk != 64 && hsk != 72) continue;   // quantized KV only at hsk 64/72
if (nr2 ==  8 && hsk != 192) continue;                              // gqa_ratio 8 skipped at hsk 128
```

So quantized KV at head size 128 — Qwen3, Llama, Mistral class, and the most common `-ctk q8_0`
configuration — has no coverage at all, and neither does head size 256; the `ncols2 = 8` GQA-batched path
is never exercised. Relaxing both takes FLASH_ATTN_EXT from 2880 to **6000** cases, **all passing on
master** — so the shapes are correct today and the generator is sound. The value is prospective: nothing
currently guards changes to the quantized-KV path at the shape most users actually run.

Happy to send that as an independent PR regardless of what happens to the kernel work.

## Questions

1. **Is `tile` dequant-on-load the direction you want?** The alternative is splitting dequant from dot
   inside `vec` and staging the dequantized K for reuse. `tile` already has the GQA batching and the SRAM
   staging, so it seemed like the smaller change — but `vec` exists for low-latency small-batch, and I may
   be missing why quantized decode belongs there.
2. **How broad should the dispatch be?** I currently prefer `tile` for symmetric q4_0/q8_0 at
   `Q->ne[0] ∈ {64, 128, 256}` whenever `gqa_opt_applies`, mirroring the f16 branch. Everything I measured
   gains, but that's three models on one GPU. Should it be narrower (e.g. gated on `gqa_ratio`)?
3. **Instantiation budget.** 6 new tile instantiations (q4_0/q8_0 × 64/128/256). Is that acceptable
   compile-time cost, or should it be trimmed?
4. **NVIDIA is unmeasured** — I only have gfx1151. The redundancy is architecture-independent but its cost
   is not; with a large L2 and high bandwidth, `vec` may still win at low `gqa_ratio`. Would someone with
   Ada/Hopper be willing to check? This is the main thing blocking a confident PR.

## How close is this to the roof now?

Fitting `t/s = B / (W + KV_bytes)` on the f16/`tile` arms (which have no dequant, so `A = 1` by
construction) gives `W = 4.04 GB` per-token weight traffic and `B = 232.4 GB/s` sustained — 91% of the
part's 256 GB/s peak — and reproduces every f16 point to within 0.3%. Measuring each format against its own
bandwidth ceiling:

| depth | format | measured | ceiling | % of roof |
| ----- | ------ | -------- | ------- | --------- |
| 32k   | f16    | 31.97    | 32.01   | **100%**  |
| 32k   | q8_0   | 38.42    | 40.42   | **95%**   |
| 32k   | q4_0   | 39.97    | 47.00   | 85%       |

**q8_0 decode at 32k goes from 41% of its bandwidth ceiling to 95%.** The defect moved quantized decode
from dequant-bound to essentially bandwidth-bound, which is where it should have been all along.

## Known limitations

- **q4_0 reaches 85% of its ceiling, not ~100%.** Two causes, both inherent to the q4_0 layout rather than
  to the GQA fix: (a) it needs two `dequantize_V` calls per 8-element chunk where q8_0 needs one, so it
  issues ~2× the unpack ALU; (b) element `j` is the low nibble of `qs[j]` and element `j+16` is the **high
  nibble of the same byte**, so with the generic `dequantize_V` interface every `qs` byte is fetched twice.
  A q4_0-specialised loader that fetches each byte once and emits both nibble halves would recover most of
  the remaining ~15%, at the cost of assuming 32-element alignment within the tile. I left it out to keep
  the change generic — happy to add it if wanted.
  (Data point for anyone tuning this: the loader is overhead-bound, not byte-bound. Forcing q4_0 to
  `ne = 2` — same bytes, 2× the chunks — costs **−31%/−38%** at 16k/32k.)
- `epc = 8` was chosen to match the q8_0 block layout, not swept.
- Only symmetric quantized KV (`K->type == V->type`); asymmetric needs `GGML_CUDA_FA_ALL_QUANTS` anyway.
- Single GPU, single vendor. All numbers are gfx1151.

## Environment

- Radeon 8060S / Ryzen AI MAX+ 395 (gfx1151, RDNA3.5, 64 GB UMA), ROCm 7.2.4.
- All measurements were taken against `4a7ee3126`; the branch has since been rebased onto upstream master,
  which required no changes (the FA sources were byte-identical across that range).
- `llama-bench -fa 1 -b 512 -ub 512 -p 512 -n 32 -d <depth> -r 2` (128k: `-r 1 --no-warmup`), page cache
  warmed, box idle, one dedicated invocation per config.
- Stock, patched and Vulkan binaries all built from that same commit with the same toolchain, run in the
  same session on the same box; arms differ only by the backend and the patch.
- The RADV arms are **vanilla upstream Vulkan** — not carrying PR #25494 or any local patches.
- Model: `Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf` (48 layers, 32 Q heads / 4 KV heads, head_dim 128).
- No rocWMMA in the build, so `ggml_cuda_should_use_wmma_fattn()` is false and the WMMA-FA branch is
  compiled out; decode dispatch would differ on a rocWMMA build.
