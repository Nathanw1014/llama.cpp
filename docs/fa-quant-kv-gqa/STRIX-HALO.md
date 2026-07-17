# Strix Halo (gfx1151) — quantized-KV flash-attention fixes for both backends

This branch carries two independent fixes that happen to be the same idea on two backends:
**dequantize quantized KV once, then reuse it**, instead of re-dequantizing it per consumer.

| Backend | Commit | What it fixes | Regime |
| ------- | ------ | ------------- | ------ |
| **Vulkan / RADV** | `vulkan : dequant q8_0 KV once in coopmat1` | the coopmat1 FA path dequantized q8_0 KV as a separate pass | prefill |
| **Vulkan / RADV** | `vulkan : fall back instead of aborting when FA scratch exceeds maxStorageBufferRange` | hard abort at long context | robustness |
| **HIP / ROCm** | `CUDA: dequantize KV on load in the tile FA kernel…` | the vec FA kernel re-dequantizes each KV head once per Q head (`gqa_ratio`×) | decode |
| — | `tests: cover quantized KV at head sizes 128 and 256, and gqa_ratio 8` | quantized KV had no correctness coverage above head size 72 | tests |

Upstream status: the Vulkan commits are llama.cpp PR #25494 (in review). The HIP commits are not yet
submitted — see `README.md` in this directory for the diagnosis, controls and open questions.

## Why one branch

On this hardware the two backends were failing the *same* way in different places. Vulkan's FA batches GQA
correctly but dequantized q8_0 KV in a separate pass, costing prefill. HIP's FA dequantizes in-kernel but
does not batch GQA, so it repeats that work `gqa_ratio` times, costing decode. Both are "dequant once and
reuse"; neither is a Strix-Halo-specific hack.

## Each fix vs its own base, both regimes

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fa-fixes-dark.svg">
  <img alt="Prefill and decode versus KV depth on gfx1151. The RADV fix lifts prefill and leaves decode unchanged; the ROCm fix lifts decode and leaves prefill unchanged." src="fa-fixes-light.svg">
</picture>

All four builds are the **same commit** (upstream master) — the only difference is the patch. Colour is the
backend, dashes are the unpatched base. The two flat lines are the point: each fix moves its own regime and
provably does not touch the other's, so they compose rather than overlap.

## Measured on gfx1151 (Radeon 8060S / Ryzen AI MAX+ 395, 64 GB UMA)

Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL (48 layers, 32 Q heads / 4 KV heads, head_dim 128), q8_0 KV.
`llama-bench -fa 1 -b 512 -ub 512 -p 512 -n 32 -d <depth> -r 2` (128k: `-r 1 --no-warmup`), page cache
warmed, box idle, one dedicated invocation per config. **All four builds are the same commit (upstream
master); only the patch differs.** A single invocation yields both metrics, so prefill and decode below are
from the same runs.

### Decode — tg32 (t/s) · the ROCm fix's regime

| depth | ROCm base | ROCm + fix | Δ | RADV base | RADV + fix | Δ |
| ----- | --------- | ---------- | - | --------- | ---------- | - |
| 0    | 52.88 | **53.74** | +1.6% | 64.73 | 66.17 | +2.2% |
| 4k   | 41.88 | **51.20** | +22.3% | 57.52 | 57.80 | +0.5% |
| 16k  | 26.37 | **44.20** | +67.6% | 48.04 | 47.78 | -0.5% |
| 32k  | 16.55 | **37.86** | +128.8% | 38.32 | 38.70 | +1.0% |
| 64k  | 8.95 | **29.40** | +228.5% | 27.94 | 27.91 | -0.1% |
| 128k | 4.68 | **19.96** | +326.5% | 18.12 | 18.05 | -0.4% |

### Prefill — pp512 (t/s) · the RADV fix's regime

| depth | ROCm base | ROCm + fix | Δ | RADV base | RADV + fix | Δ |
| ----- | --------- | ---------- | - | --------- | ---------- | - |
| 0    | 947.4 | 955.6 | +0.9% | 959.8 | **989.2** | +3.1% |
| 4k   | 836.5 | 833.3 | -0.4% | 647.4 | **735.9** | +13.7% |
| 16k  | 506.3 | 508.9 | +0.5% | 333.1 | **443.0** | +33.0% |
| 32k  | 310.8 | 310.9 | +0.1% | 202.8 | **285.4** | +40.7% |
| 64k  | 183.8 | 183.8 | +0.0% | 112.3 | **168.7** | +50.2% |
| 128k | 93.21 | 103.0 | +10.5% | 57.59 | **89.79** | +55.9% |

The ROCm prefill row at 128k (+10.5%) is **noise, not a win**: every other depth is flat within ±0.9%, the
128k arm is a single shot (`-r 1`), and that arm was separately measured to swing ~±9% run to run. The ROCm
change cannot move prefill at head_dim 128 — prefill takes the `mma` path, which the patch does not touch.
Treat it as flat.

Read the Δ columns together: the ROCm fix moves decode and leaves prefill flat; the RADV fix moves prefill
and leaves decode flat. Neither is a tuning knob traded against the other — they compose.

Two things the decode table says that the headline number does not. RADV still wins shallow context, and
the crossover where ROCm overtakes it is around 32k. And the ROCm fix's value grows with depth, because the
redundant dequant it removes scales with the KV cache.

## What to run on this hardware

- **Long-context decode (agentic, deep KV): ROCm + `-ctk q8_0 -ctv q8_0`** — wins past ~32k with this branch.
  Without the fix this was the worst config available; it is now competitive.
- **Shallow context: RADV + q8_0** — still ahead below ~32k.
- **Prefill-heavy ingest: RADV + q8_0** benefits from the coopmat1 dequant-once commit.
- Past ~96k context on a 64 GB box, q8_0 KV is effectively mandatory for host headroom regardless of backend.

## Verification

- `test-backend-ops -o FLASH_ATTN_EXT`: 6000/6000 against the CPU reference on the HIP backend, with the
  quantized-KV coverage this branch adds (the stock suite ran 2880 and never exercised quantized KV above
  head size 72).
- Greedy generation (temp 0) is byte-identical between the stock and patched HIP paths.
- f16 is unchanged on both backends; HIP prefill is unaffected at head_dim 64/128.

## Caveats

- All numbers are gfx1151. NVIDIA is unmeasured for the HIP change; the redundancy is architecture-
  independent but its cost is not.
- The HIP dispatch prefers `tile` for symmetric q4_0/q8_0 whenever the GQA optimization applies. That
  boundary is measured on three models on one GPU, not settled.
- q4_0 reaches ~85% of its bandwidth ceiling vs q8_0's ~95%; see `README.md` for why.
- This branch is a convenience for Strix Halo users. The upstreamable units are the individual commits.
