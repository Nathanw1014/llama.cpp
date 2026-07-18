ggml_cuda_init: found 1 ROCm devices (Total VRAM: 59392 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 59392 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q5_K - Medium |  24.76 GiB |    34.66 B | ROCm       |  -1 |   1 | pp2048 @ d32768 |        538.68 ± 2.21 |
| qwen35moe 35B.A3B Q5_K - Medium |  24.76 GiB |    34.66 B | ROCm       |  -1 |   1 |   tg32 @ d32768 |         38.52 ± 1.65 |

build: unknown (0)
