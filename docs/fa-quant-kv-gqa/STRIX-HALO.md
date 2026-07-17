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

## Measured on gfx1151 (Radeon 8060S / Ryzen AI MAX+ 395, 64 GB UMA)

Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL (48 layers, 32 Q heads / 4 KV heads, head_dim 128), q8_0 KV,
`llama-bench -fa 1 -b 512 -ub 512 -p 512 -n 32 -d <depth> -r 2`. All arms same box, same session, same
model file, GPU otherwise idle.

Decode (tg32, t/s):

| depth | ROCm stock | ROCm + fix | RADV (vanilla) |
| ----- | ---------- | ---------- | -------------- |
| 0     | 53.89      | 55.14      | 65.45          |
| 4k    | 42.25      | 51.45      | 56.77          |
| 16k   | 26.67      | **44.59**  | 47.55          |
| 32k   | 16.74      | **37.97**  | 38.25          |
| 64k   | 8.95       | **29.69**  | 27.83          |
| 128k  | 4.68       | **19.77**  | 17.85          |

The HIP fix is worth **+322% at 128k**, and moves ROCm from 3.1× behind RADV at 64k to ahead of it. The
crossover is around 32k: **RADV is still the better choice for shallow context** (65.45 vs 55.14 at depth 0).

Prefill on RADV (pp512, t/s) — what the coopmat1 dequant-once commit buys, measured on this branch against
vanilla Vulkan at the same commit:

| depth | RADV vanilla | RADV + dequant-once |
| ----- | ------------ | ------------------- |
| 16k   | 334.84       | **444.88** (+32.9%) |
| 32k   | 201.95       | **284.16** (+40.7%) |

So the two commits are complementary rather than overlapping: the Vulkan one buys prefill on RADV, the HIP
one buys decode on ROCm. Together they remove the quantized-KV penalty from both halves of a long-context
session on this hardware.

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
