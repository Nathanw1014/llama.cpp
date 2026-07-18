#!/bin/bash
# Matched re-run to reconcile absolute t/s against thefrontierlab.ai numbers.
# Their model class (Qwen3.6-35B-A3B Q5), their batch config (2048/512), FA on, 32k depth.
# The key control is our f16 arm: if it lands near their f16 (ROCm 41.5 tg / 567 pp, Vulkan 46.7 / 670),
# that proves the earlier "ours looks less" gap was model+quant+batch, not a slow backend.
#
# Protocol (3 windows were invalidated on this box before): pre-warm file, GTT<1GB before each arm,
# >=25s settle between arms, end-canary re-run of arm 1 to prove the window held.
set -u
OUT=/mnt/data/strix-combined-wt/matched-results
mkdir -p "$OUT"
M=/home/alloy/models/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
MD=/models/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
FLAGS="-fa 1 -b 2048 -ub 512 -p 2048 -n 32 -d 32768 -r 2 -o md"
DK="docker run --rm --oom-score-adj=600 --device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined -v /home/alloy/models:/models:ro"

gtt() { awk '{printf "%.2f",$1/1073741824}' /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1; }
wait_gtt() { for i in $(seq 1 30); do g=$(gtt); if awk "BEGIN{exit !($g<1.5)}"; then return 0; fi; sleep 5; done; }

echo "pre-warming model file..."; cat "$M" > /dev/null

# tag  runner(hip|vk)  worktree  kvflags
run() {
  local tag=$1 kind=$2 wt=$3 kv=$4
  wait_gtt
  echo "MARKER-START $tag  (GTT=$(gtt) GB)"
  if [ "$kind" = "hip" ]; then
    $DK -v "$wt":/work -w /work tile-build-env /work/build-hip/bin/llama-bench -m $MD $kv $FLAGS > "$OUT/$tag.md" 2>&1
  else
    "$wt"/build-vulkan/bin/llama-bench -m "$M" $kv $FLAGS > "$OUT/$tag.md" 2>&1
  fi
  echo "MARKER-DONE  $tag  pp=$(grep -oE 'pp2048[^|]*\|[ ]*[0-9.]+' "$OUT/$tag.md"|grep -oE '[0-9.]+$') tg=$(grep -oE 'tg32[^|]*\|[ ]*[0-9.]+' "$OUT/$tag.md"|grep -oE '[0-9.]+$')"
  sleep 25
}

run rocm-base-f16  hip /mnt/data/mbase              ""
run rocm-base-q8   hip /mnt/data/mbase              "-ctk q8_0 -ctv q8_0"
run rocm-fix-q8    hip /mnt/data/strix-combined-wt  "-ctk q8_0 -ctv q8_0"
run radv-base-f16  vk  /mnt/data/mbase              ""
run radv-base-q8   vk  /mnt/data/mbase              "-ctk q8_0 -ctv q8_0"
run radv-fix-q8    vk  /mnt/data/strix-combined-wt  "-ctk q8_0 -ctv q8_0"
# end-canary: repeat arm 1; if it matches, the window held
run rocm-base-f16-canary hip /mnt/data/mbase ""

touch "$OUT/MATCHED-DONE"; echo "ALL DONE"
