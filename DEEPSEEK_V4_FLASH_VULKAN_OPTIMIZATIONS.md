# DeepSeek V4 Flash Vulkan Optimizations

## Overview

This branch improves DeepSeek V4 Flash prefill performance on AMD Strix Halo
using the Vulkan backend. The work targets the three operations that dominated
long-context prefill:

1. Lightning Indexer scoring
2. Indexed sparse Flash Attention
3. Mixture-of-Experts `MUL_MAT_ID` dispatch

The implementation keeps scalar or existing dense paths available when the
specialized path is unsupported or would not be profitable.

Test system:

- GPU: AMD Radeon 8060S, Strix Halo / GFX1151
- Vulkan driver: RADV
- Subgroup size: 64
- Cooperative matrices: `VK_KHR_cooperative_matrix`
- Model: DeepSeek V4 Flash IQ3_XXS, 97.05 GiB, four GGUF shards
- Branch: `deepseek-v4-flash-strix-halo`
- Initial base commit: `a18067a85`
- Optimization commit: `725922188`

## Final Prefill Results

The final context-depth sweep used a 512-token prompt, one measured repetition,
memory mapping disabled, and no warm-up:

```bash
./build-vulkan-radv/bin/llama-bench \
    -m /path/to/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
    -n 0 \
    -d 0,10000,30000,60000,90000 \
    -mmp 0 \
    -r 1 \
    --no-warmup \
    --progress
```

| Context depth | Prefill speed |
| ---: | ---: |
| 0 | 157.05 tokens/s |
| 10,000 | 134.86 tokens/s |
| 30,000 | 121.53 tokens/s |
| 60,000 | 115.57 tokens/s |
| 90,000 | 109.47 tokens/s |

At a context depth of 32,768, performance improved from the original 87.87
tokens/s to 123.06 tokens/s. This is an improvement of approximately 40%.

The complete sweep output is stored in:

```text
deepseek-v4-flash-iq3_xxs-depth-sweep-0-10000-30000-60000-90000.log
```

## Lightning Indexer

### Original bottleneck

The original Vulkan implementation used a scalar shader for Lightning Indexer
scoring. At a context depth of 32K, this operation consumed approximately 714
ms during one 512-token prefill.

### Prefill cooperative-matrix path

`lightning_indexer_cm.comp` processes a 16-key by 16-token tile using KHR
cooperative matrices. It is selected when the token batch contains at least 16
tokens.

The tiled path increased the Lightning Indexer prefill microbenchmark from
approximately 2.12 TFLOPS to 5.76 TFLOPS. In the 32K model profile, the
operation decreased from approximately 714 ms to 247 ms.

### Decode cooperative-matrix path

`lightning_indexer_decode_cm.comp` uses a 16-key by 16-head arrangement for
single-token decode. This avoids applying a prefill-oriented tile to decode.

Representative microbenchmark results:

| KV length | Scalar | Cooperative matrix |
| ---: | ---: | ---: |
| 256 | 28.8 us | 5.8 us |
| 512 | 29.0 us | 5.8 us |
| 4,096 | 40.3 us | 12.8 us |

Decode model throughput changed only slightly because MoE matmuls, rather than
Lightning Indexer, dominate full-model decode.

### Dispatch policy

The Vulkan backend selects the Lightning Indexer pipeline as follows:

| Token batch | Pipeline |
| ---: | --- |
| 1 | Decode cooperative-matrix shader |
| 2 to 15 | Scalar subgroup shader |
| 16 or more | Prefill cooperative-matrix shader |

The scalar implementation remains the fallback when cooperative matrices are
not available. Partial tiles and multi-stream shapes are supported.

## Indexed Sparse Flash Attention

### Metadata propagation

DeepSeek V4 Flash already computes top-k key indices with its Lightning
Indexer. The change propagates these indices to `GGML_OP_FLASH_ATTN_EXT`
instead of converting sparse selection into only a dense mask.

The API entry point is:

```c
ggml_flash_attn_ext_add_top_k(flash_attn, top_k, n_kv_raw);
```

The metadata is attached in the model graph and consumed by the Vulkan
backend. Other models and attention operations without this metadata continue
to use their existing paths.

### Sparse shader

`flash_attn_top_k.comp` reads:

- Query vectors
- Raw key vectors
- Value vectors
- Top-k key indices
- The attention mask

It computes attention only for selected keys instead of scanning the full
dense KV range. The current DeepSeek configuration uses up to 512 selected
keys.

The sparse path is used only when the dense KV length is at least three times
the active top-k length:

```text
dense_kv >= 3 * active_kv
```

This threshold avoids dispatching the sparse implementation when its index
handling overhead is unlikely to recover enough work. Unsupported tensor
layouts or missing metadata fall back to the existing dense Flash Attention
path.

At 32K context, indexed sparse attention reduced the measured long-context
attention time from approximately 1.596 seconds to 611 ms.

### Rejected cooperative-matrix prototype

A single-subgroup cooperative-matrix version of sparse attention was tested
but not retained. At 16K context it achieved 120.44 tokens/s, compared with
123.03 tokens/s for the scalar sparse shader. The prototype was removed and is
not part of the committed code.

Further cooperative-matrix work should adapt the existing multi-subgroup dense
Flash Attention structure rather than recreate the rejected single-subgroup
design.

## MoE `MUL_MAT_ID` Dispatch

### Problem

The Vulkan backend previously selected a matrix multiplication pipeline using
the total token count. This is inaccurate for MoE operations because each
expert processes only a fraction of the routed tokens.

For the tested model shape:

```text
tokens per expert = ceil(512 tokens * 6 selected experts / 256 experts)
                  = 12 tokens per expert
```

Selecting a pipeline as if every expert processed all 512 tokens chose a tile
that was too large for the actual workload.

### Fix

Pipeline selection and alignment now use average routed tokens per expert:

```cpp
const uint32_t n_per_expert = CEIL_DIV(nei0 * nei1, n_as);
```

The real operation dimensions and output remain unchanged. Only the pipeline
selection estimate is corrected.

### Microbenchmark results

The exact DeepSeek-shaped test cases use 256 experts, top-6 routing, and a
512-token batch.

| Weight type and shape | Before | After |
| --- | ---: | ---: |
| IQ2_XS, 2048 x 4096 | 3.16 TFLOPS | 3.73 TFLOPS |
| IQ2_XS, 4096 x 2048 | 3.17 TFLOPS | 3.75 TFLOPS |
| IQ3_XXS, 2048 x 4096 | 2.86 TFLOPS | 3.25 TFLOPS |
| IQ3_XXS, 4096 x 2048 | 2.91 TFLOPS | 3.63 TFLOPS |

A forced medium tile was also measured and returned performance to roughly
3.16 TFLOPS for IQ2_XS and 2.86 TFLOPS for IQ3_XXS. The effective-load-based
small tile is therefore retained.

## Source Files

| File | Purpose |
| --- | --- |
| `ggml/include/ggml.h` | Declares top-k Flash Attention metadata API |
| `ggml/src/ggml.c` | Attaches top-k tensors and raw KV length to attention nodes |
| `src/llama-graph.h` | Extends attention graph-builder inputs |
| `src/llama-graph.cpp` | Propagates optional top-k metadata into Flash Attention |
| `src/models/deepseek4.cpp` | Connects DeepSeek Lightning Indexer top-k output to attention |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Creates and dispatches new pipelines; fixes MoE pipeline selection |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k.comp` | Indexed sparse attention shader |
| `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer.comp` | Scalar Lightning Indexer fallback |
| `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp` | Cooperative-matrix prefill shader |
| `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_decode_cm.comp` | Cooperative-matrix decode shader |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | Registers shaders for embedding |
| `tests/test-backend-ops.cpp` | Lightning Indexer and model-shaped MoE tests |

## Validation

The retained implementation was checked with:

- Nine Lightning Indexer CPU/Vulkan comparison cases
- Partial-tile Lightning Indexer shapes
- Multi-stream Lightning Indexer shapes
- IQ2_XS and IQ3_XXS model-shaped `MUL_MAT_ID` correctness cases
- End-to-end DeepSeek V4 Flash prefill benchmarks
- Vulkan build of `llama-bench` and `test-backend-ops`
- `git diff --check`

The final measured test runs passed all selected backend correctness cases.

## Limitations and Follow-up Work

- Indexed sparse Flash Attention currently has end-to-end model validation but
  does not yet have a dedicated, numerically sensitive backend parity test.
- The specialized attention shader currently targets the tensor shapes and F16
  KV layout used by DeepSeek V4 Flash. Unsupported cases use dense attention.
- A faster sparse-attention implementation may be possible by adapting the
  existing multi-subgroup Flash Attention kernel to indexed key access.
- After the current improvements, MoE matmuls and sparse attention remain the
  main candidates for additional prefill optimization.
- Benchmark runs used one repetition and no warm-up. Multi-run measurements
  should be used when comparing small changes.

## Reproducing the Backend Tests

After configuring a Vulkan build, the relevant tests can be selected with
`test-backend-ops`. For example:

```bash
./build-vulkan-radv/bin/test-backend-ops test \
    -b Vulkan0 \
    -o LIGHTNING_INDEXER
```

The model-shaped MoE performance cases can be run with:

```bash
./build-vulkan-radv/bin/test-backend-ops perf \
    -b Vulkan0 \
    -o MUL_MAT_ID \
    -p 'n_mats=256,n_used=6,.*m=(2048|4096),n=512'
```

Because this repository directory was renamed after the original build, an
existing CMake cache may contain stale absolute paths. Reconfigure the build
directory before rebuilding if CMake reports the old repository path.
