# Benchmark data — raw `llama-bench` receipts behind STRIX-HALO.md

Every percentage in [STRIX-HALO.md](STRIX-HALO.md) traces to the raw `llama-bench` output below, error bars included. One invocation per (config, depth) emitted both a `pp512` (prefill) and a `tg32` (decode) row, so prefill and decode share the same run.

## Setup

- **Box:** Radeon 8060S / Ryzen AI MAX+ 395 (gfx1151, RDNA3.5, 64 GB UMA)
- **Model:** `Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf`  
  sha256 `469bdf01f100697bcc7f40483394dc4db4409b45e58b06275af29e4a9d4fe6bb` — 48 layers, 32 Q / 4 KV heads, head_dim 128
- **Command:** `llama-bench -fa 1 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 -p 512 -n 32 -d <depth> -r 2` (128k: `-r 1 --no-warmup`)
- **Window:** page cache warmed, box otherwise idle, one dedicated invocation per config; all four builds are the same source commit and only the patch differs.
- **Build identity:** the RADV binaries self-report their commit (`635cdd5` unpatched base, `f1d5983` this branch). The HIP binaries emit `build: unknown (0)` because git was unavailable in the build container; they were compiled from the same two trees — unpatched master `635cdd5` (base) and this branch (fix) — with an identical ROCm 7.2.4 / gfx1151 toolchain.

## Raw output

### rocm-base — ROCm, unpatched master

| depth | pp512 (t/s) | tg32 (t/s) |
| ----- | ----------- | ---------- |
| 0     | 947.37 ± 40.98 | 52.88 ± 0.66 |
| 4k    | 836.45 ± 3.91 | 41.88 ± 0.23 |
| 16k   | 506.30 ± 8.83 | 26.37 ± 0.17 |
| 32k   | 310.76 ± 1.16 | 16.55 ± 0.06 |
| 64k   | 183.82 ± 0.18 | 8.95 ± 0.02 |
| 128k  | 93.21 ± 0.00 | 4.68 ± 0.00 |

### rocm-fix — ROCm, this branch

| depth | pp512 (t/s) | tg32 (t/s) |
| ----- | ----------- | ---------- |
| 0     | 955.61 ± 14.15 | 53.74 ± 1.81 |
| 4k    | 833.32 ± 0.75 | 51.20 ± 0.59 |
| 16k   | 508.86 ± 9.79 | 44.20 ± 0.46 |
| 32k   | 310.95 ± 0.16 | 37.86 ± 0.40 |
| 64k   | 183.85 ± 0.41 | 29.40 ± 0.35 |
| 128k  | 103.00 ± 0.00 | 19.96 ± 0.00 |

### radv-base — RADV, unpatched master

| depth | pp512 (t/s) | tg32 (t/s) |
| ----- | ----------- | ---------- |
| 0     | 959.79 ± 4.16 | 64.73 ± 1.74 |
| 4k    | 647.39 ± 13.95 | 57.52 ± 0.45 |
| 16k   | 333.14 ± 3.52 | 48.04 ± 0.28 |
| 32k   | 202.80 ± 4.12 | 38.32 ± 0.05 |
| 64k   | 112.31 ± 1.21 | 27.94 ± 0.10 |
| 128k  | 57.59 ± 0.00 | 18.12 ± 0.00 |

### radv-fix — RADV, this branch

| depth | pp512 (t/s) | tg32 (t/s) |
| ----- | ----------- | ---------- |
| 0     | 989.21 ± 0.32 | 66.17 ± 0.32 |
| 4k    | 735.93 ± 17.72 | 57.80 ± 0.95 |
| 16k   | 443.03 ± 5.18 | 47.78 ± 0.08 |
| 32k   | 285.39 ± 3.32 | 38.70 ± 0.33 |
| 64k   | 168.65 ± 1.68 | 27.91 ± 0.13 |
| 128k  | 89.79 ± 0.00 | 18.05 ± 0.00 |

## Derived percentages (as they appear in STRIX-HALO.md)

### Token generation — ROCm fix vs its unpatched default (tg32)

| depth | ROCm base | ROCm + fix | change |
| ----- | --------- | ---------- | ------ |
| 0     | 52.88 ± 0.66 | 53.74 ± 1.81 | **+1.6%** |
| 4k    | 41.88 ± 0.23 | 51.20 ± 0.59 | **+22.3%** |
| 16k   | 26.37 ± 0.17 | 44.20 ± 0.46 | **+67.6%** |
| 32k   | 16.55 ± 0.06 | 37.86 ± 0.40 | **+128.8%** |
| 64k   | 8.95 ± 0.02 | 29.40 ± 0.35 | **+228.5%** |
| 128k  | 4.68 ± 0.00 | 19.96 ± 0.00 | **+326.5%** |

### Prefill — RADV fix vs its unpatched default (pp512)

| depth | RADV base | RADV + fix | change |
| ----- | --------- | ---------- | ------ |
| 0     | 959.79 ± 4.16 | 989.21 ± 0.32 | **+3.1%** |
| 4k    | 647.39 ± 13.95 | 735.93 ± 17.72 | **+13.7%** |
| 16k   | 333.14 ± 3.52 | 443.03 ± 5.18 | **+33.0%** |
| 32k   | 202.80 ± 4.12 | 285.39 ± 3.32 | **+40.7%** |
| 64k   | 112.31 ± 1.21 | 168.65 ± 1.68 | **+50.2%** |
| 128k  | 57.59 ± 0.00 | 89.79 ± 0.00 | **+55.9%** |

### Off-regime controls (should be flat)

| depth | ROCm pp512 base→fix | RADV tg32 base→fix |
| ----- | ------------------- | ------------------ |
| 0     | +0.9% | +2.2% |
| 4k    | -0.4% | +0.5% |
| 16k   | +0.5% | -0.5% |
| 32k   | +0.1% | +1.0% |
| 64k   | +0.0% | -0.1% |
| 128k  | +10.5% | -0.4% |

The ROCm prefill column and the RADV decode column are the off-regime halves: near-zero at every depth (the ROCm pp512 +10.5% at 128k is single-shot `-r 1` noise — see STRIX-HALO.md). Each fix moves only its own regime.
