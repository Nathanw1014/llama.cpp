ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q5_K - Medium |  24.76 GiB |    34.66 B | Vulkan     |  -1 |   q8_0 |   q8_0 |   1 | pp2048 @ d32768 |        572.60 ± 0.39 |
| qwen35moe 35B.A3B Q5_K - Medium |  24.76 GiB |    34.66 B | Vulkan     |  -1 |   q8_0 |   q8_0 |   1 |   tg32 @ d32768 |         47.56 ± 0.50 |

build: f1d5983 (193)
