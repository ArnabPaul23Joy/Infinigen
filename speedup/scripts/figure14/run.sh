#!/usr/bin/env bash
# ─── KAGGLE-COMPATIBLE figure14/run.sh ───────────────────────────────────────
# Differences from the original lab script:
#   * Paths are resolved relative to THIS script, not $PWD, so it runs from any
#     working directory (Kaggle starts you in /kaggle/working).
#   * Dependencies are installed on first run (set INFINIGEN_INSTALL=0 to skip).
#   * Model / batch / sequence lengths default to values that fit a single
#     16 GB Kaggle GPU (T4 or P100). The paper used a 48 GB A6000 with OPT-13B,
#     batch 20, prompt 1920 — that OOMs on Kaggle. Override with env vars to
#     scale back up on bigger hardware.
#   * The UVM section is left out: it needs nvcc/g++ against $CUDA_HOME and a
#     custom CUDAPluggableAllocator, which is not practical in a Kaggle kernel.
#
# Usage on Kaggle (in a code cell):
#   !bash "/kaggle/working/InfiniGen/speedup/scripts/figure14/run.sh"
# Or to push toward the paper config on a bigger GPU:
#   !MODEL=huggingface/opt-13b GPU_BATCH=20 PROMPT_LEN=1920 MAX_NUM_KV=400 \
#       bash .../run.sh
set -euo pipefail

# ─── Resolve repo layout relative to this script ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SPEEDUP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"     # .../InfiniGen/speedup
FLEXGEN_PATH="$SPEEDUP_DIR/flexgen"
INFINIGEN_PATH="$SPEEDUP_DIR/infinigen"

# ─── One-time setup (safe to re-run; set INFINIGEN_INSTALL=0 to skip) ─────────
# flexgen pins only torch>=1.12 / transformers>=4.24, so this will NOT downgrade
# the torch that Kaggle ships with.
if [ "${INFINIGEN_INSTALL:-1}" = "1" ]; then
  pip install -q -e "$INFINIGEN_PATH" -e "$FLEXGEN_PATH"
fi

# ─── Config (override any of these via environment variables) ────────────────
MODEL="${MODEL:-huggingface/opt-1.3b}"
# weights / KV cache / activations split across GPU,CPU,disk.
# "100 0 0 100 100 0" = everything on GPU except the KV cache, which goes to
# CPU RAM — the offloading scenario InfiniGen targets, kept small for 16 GB.
PERCENT="${PERCENT:-100 0 0 100 100 0}"
GPU_BATCH="${GPU_BATCH:-4}"
PROMPT_LEN="${PROMPT_LEN:-512}"
GEN_LEN="${GEN_LEN:-128}"
# InfiniGen knobs (see paper Sec. on speculation):
ALPHA="${ALPHA:-4}"                        # speculation threshold
PARTIAL_WEIGHT_RATIO="${PARTIAL_WEIGHT_RATIO:-0.2}"  # frac of Q/K cols for speculation
MAX_NUM_KV="${MAX_NUM_KV:-128}"            # upper bound on KV tokens fetched/step

INPUT_PATH="$SCRIPT_DIR/pg19_firstbook.txt"

# ─── FlexGen path: swap in the InfiniGen implementation via symlink ──────────
# Only "infinigen" by default; add "original" "int4" "h2o" to also run baselines.
for SCHEME in "infinigen"
do
  # -f so it doesn't error when the symlink is missing (fresh checkout / re-run).
  rm -f "$FLEXGEN_PATH/flexgen/flex_opt.py"
  rm -f "$FLEXGEN_PATH/flexgen/pytorch_backend.py"

  if [ "$SCHEME" = "int4" ]; then
    # int4 reuses the original baseline code + the --compress-cache flag below.
    ln -s "../original/flex_opt.py"        "$FLEXGEN_PATH/flexgen/flex_opt.py"
    ln -s "../original/pytorch_backend.py" "$FLEXGEN_PATH/flexgen/pytorch_backend.py"
  else
    ln -s "../$SCHEME/flex_opt.py"        "$FLEXGEN_PATH/flexgen/flex_opt.py"
    ln -s "../$SCHEME/pytorch_backend.py" "$FLEXGEN_PATH/flexgen/pytorch_backend.py"
  fi

  CMD="--model $MODEL --percent $PERCENT --overlap false"
  CMD="$CMD --gpu-batch-size $GPU_BATCH --num-gpu-batches 1"
  CMD="$CMD --prompt-len $PROMPT_LEN --gen-len $GEN_LEN"
  CMD="$CMD --warmup-input-path $INPUT_PATH --test-input-path $INPUT_PATH"

  if [ "$SCHEME" = "int4" ]; then
    CMD="$CMD --compress-cache"
  elif [ "$SCHEME" = "h2o" ]; then
    CMD="$CMD --max-num-kv $MAX_NUM_KV --hh-ratio 0.1 --hh-all"
  elif [ "$SCHEME" = "infinigen" ]; then
    CMD="$CMD --alpha $ALPHA --partial-weight-ratio $PARTIAL_WEIGHT_RATIO --max-num-kv $MAX_NUM_KV"
  fi

  echo "=== Running scheme: $SCHEME ==="
  echo "    python -m flexgen.flex_opt $CMD"
  ( cd "$FLEXGEN_PATH" && python -m flexgen.flex_opt $CMD )
done
