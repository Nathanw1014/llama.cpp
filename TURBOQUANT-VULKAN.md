# TurboQuant KV cache on Vulkan (RADV / gfx1151)

Experimental branch adding TurboQuant KV-cache quantization (`tq3_0`, `tq4_0`) to the
CPU and Vulkan backends, developed and measured on a Radeon 8060S (RDNA 3.5, gfx1151).

TurboQuant (Zandieh et al., [arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
rotates each head-dim block by a randomized Walsh-Hadamard transform, scales by the
per-block RMS, quantizes against a Lloyd-Max Gaussian codebook, and stores a 1-bit
residual sign. `QK_TQ = 64`, so a head-dim row is 1-4 blocks.

| type | block | bits/val | vs fp16 |
|---|---|---|---|
| `tq3_0` | 26 B / 64 vals | 3.25 | 4.92x |
| `tq4_0` | 34 B / 64 vals | 4.25 | 3.76x |

## Status

Working end-to-end on Vulkan, hardware-verified. Both types, read and write paths.

**Use `tq4_0` for K.** `tq3_0` is fine as V but collapses as K — output degrades to
nonsense regardless of what V is, and the same degradation reproduces on the CPU
backend, so it is a property of the format at 3.25 bits, not a backend defect.
The recommended configuration is `-ctk tq4_0 -ctv tq3_0`, which is both smaller and
marginally faster than `tq4_0/tq4_0`.

## Why previous Vulkan attempts failed

There are two independent walls, one per direction. Issue
[#22842](https://github.com/ggml-org/llama.cpp/issues/22842) reports both symptoms
("turbo3 KV produces garbage output, turbo4/SET_ROWS crashes").

**Read — garbage output.** The Vulkan FA dequant interface is
`dequantize4(ib, iqs, ...)`, returning 4 coordinates whose values depend only on their
own quants plus a block scale. TurboQuant's inverse RHT couples all 64 coordinates, so
4 outputs cannot be produced without reconstructing the whole block. Forcing a block
transform through a per-element interface is what produces noise.

**Write — hard crash.** The KV cache is filled by `GGML_OP_SET_ROWS`, which on Vulkan
quantizes through `copy_to_quant.comp`, compiled once per destination type. A new KV
type needs four separate registrations (supports_op gate, pipeline registration,
shader generation, and the encoder body). Miss any one and `supports_op` returns false
— and because the KV cache is *pre-allocated* in the Vulkan buffer, the scheduler
cannot fall back to CPU, so it aborts with
`cache_k_l0 cannot run the operation (SET_ROWS)`.

The write path is easy to miss because the interesting work is on the read side, and
because pre-quantizing a tensor and running FA on it never exercises the encoder.

## Performance

Llama-3.2-1B (hd64), Vulkan/RADV gfx1151, depth 16384, r=5:

| KV (K/V) | tg32 t/s | pp512 t/s | bits/val |
|---|---:|---:|---:|
| `f16/f16` | 105.48 ± 1.78 | 1858.79 ± 27.78 | 16.00 |
| `q8_0/q8_0` | **109.14 ± 2.32** | **2339.39 ± 94.97** | 8.50 |
| `tq4_0/tq4_0` | 39.84 ± 0.16 | 446.31 ± 2.35 | 4.25 |
| `tq4_0/tq3_0` | 40.82 ± 0.28 | 468.51 ± 5.30 | 3.75 |

**TurboQuant is not a speed feature.** `q8_0` is faster than both f16 and TurboQuant
at depth *and* is half the size of f16, so it is the right default for most workloads.
The randomized Hadamard dequant is compute-bound and costs roughly 2.7x decode and 5x
prefill against `q8_0` here.

The value is capacity. On a 64 GB unified-memory box (58 GiB GTT ceiling),
GLM-4.5-Air Q3_K_S at its full 128k context:

| KV | footprint | result |
|---|---:|---|
| `q8_0/q8_0` | 61.5 GiB | `amdgpu: Not enough memory for command submission` |
| `tq4_0/tq4_0` | 55.2 GiB | runs — 23.7 t/s prompt, 21.3 t/s generation |

That is the case TurboQuant exists for: it is the difference between running a model at
a better weight quant with full context and not running it at all.

## Optimizations

* **Cooperative once-per-block dequant.** `dequantize4` recomputes the 64-coord inverse
  RHT for every 4 coordinates (16x redundant). The coopmat1 staging fill instead
  dequantizes each block once into shared memory with a cooperative FWHT. Worth +86.7%
  at 32k depth, and the advantage grows with depth.
* **Wider query tiles.** FA re-stages the KV tile per *query* tile, so TurboQuant pays a
  full block-RHT per tile. The coopmat1 QK/PV matmuls now iterate `Br/MatBr` row-tiles
  with the K/V loads hoisted out, letting one dequantized tile serve more query rows:
  +46% prefill at hd64, no decode cost. Tile width is clamped to what shared memory
  allows and restricted to small head dims, because a wider tile also grows the FA
  device-memory workspace and can push a tight model over the memory ceiling.

## Validation

* `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0` — 10183/10183, including tq3_0/tq4_0
  at hs=64 and 128, GQA, non-block-multiple KV, mixed with f16, and both f32 and f16
  accumulator precision.
* `test-backend-ops -o SET_ROWS -b Vulkan0` — 343/343, including TurboQuant
  destinations at 64/128/256-wide rows (the GPU encoder had no coverage before).
* A byte-exact numpy/C reference for the shipped QK64/fp16 format reproduces the
  deployed `quantize_row_tq{3,4}_0_ref` output for 256/256 blocks of both types.

## Known limitations

* `tq3_0` as K is unusable (see above). `tq4_0` K is required.
* Head dims must be a multiple of 64.
* coopmat2 is unavailable on RDNA 3.5, so this targets the coopmat1 and scalar paths.
* Not upstreamable as-is; this is a private experimental branch.
