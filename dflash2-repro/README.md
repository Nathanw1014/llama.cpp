# DFlash2 capability-gate repro

Reproduces the trap fixed by "spec : capability-gate the DFlash2 draft path":
`is_dflash2` was armed from the `dflash.selector_top_k` metadata key alone, so a
draft GGUF carrying the key without a decode graph that builds the selector
lattice made the host read `h_nextn` hidden states as a lattice.

Both scripts copy a public draft GGUF and inject the mismatch. They need
`numpy` and the in-tree `gguf-py`.

## 1. key without lattice (silent performance poison)

```
python3 tamper_key_only.py <dflash-v1-draft.gguf> tampered.gguf
```

Input: any DFlash **v1** draft (e.g. the incoai Qwen3.8-27B v1 draft). Run the
output as `-md` with `--spec-type draft-dflash` against its target model.

- before the fix: no crash, no diagnostic. The v1 graph writes hidden states
  into `t_h_nextn`, the host decodes them as candidate ids, and verification
  rejects everything. Measured on Qwen3.8-27B / gfx1151-Vulkan: 0/1556 drafts
  accepted, 4.77 t/s vs ~11.4 t/s without speculation.
- after the fix: warning at startup, clean fallback to the v1 draft path
  (same setup: 162/295 accepted, 25.6 t/s).

## 2. selector tensor on the DSV4 backbone (load-time refusal)

```
python3 inject_selector_tensor.py <dspark-dsv4-draft.gguf> injected.gguf
```

Input: any DSV4-backbone DSpark draft (e.g. the public DeepSeek-V4-Flash DSpark
sidecar). Load the output with any llama.cpp tool.

- before the fix: fails load with a misleading "missing conv/selector
  metadata" error (or, with full selector metadata, loads into the runtime
  trap above: `graph_dsv4` has no `build_post_sampling`).
- after the fix: "DFlash2 selector is not supported on the DSV4 backbone".
