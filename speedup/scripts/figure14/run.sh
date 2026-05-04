# ─── SECTION 1: UVM PATH ─────────────────────────────────────────────────────
# Tests two UVM-based schemes on a single fixed config (OPT-13B equivalent,
# batch=20, prompt=1920, gen=128). UVM (Unified Virtual Memory) lets the KV
# cache exceed GPU VRAM by transparently paging between VRAM and CPU DRAM.

UVM_PATH=$PWD/../../uvm
# LD_LIBRARY_PATH lets Python find allocate.so (the custom CUDA allocator)
# at runtime when transformer.py calls torch.cuda.memory.CUDAPluggableAllocator
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

for SCHEME in "uvm" "uvm_h2o"
do
  # Compile the custom CUDA allocator that replaces cudaMalloc with
  # cudaMallocManaged, giving all tensors a unified CPU+GPU address space.
  # Must be recompiled each scheme run because the .so is deleted afterwards.
  g++ $UVM_PATH/allocate.cpp -o allocate.so --shared -fPIC -I$CUDA_HOME/include

  # Model config matching OPT-13B: embed_dim=5120, 40 heads, 40 layers.
  # do_layer_norm_before = OPT's pre-norm style (used in 125M, 1.7B...175B).
  CMD="--embed_dim 5120 --ffn_dim 20480 --enable_bias --n_head 40 --do_layer_norm_before --n_layer 40 --bsz 20 --prompt_len 1920 --gen_len 128 --runs 1"

  if [ "$SCHEME" = "uvm_h2o" ]
  then
    # H2O (Heavy Hitter Oracle): after prefill, prune the KV cache to keep only
    # the top 20% of tokens by accumulated attention score. This bounds the UVM
    # working set to a fixed size regardless of sequence length.
    CMD=$CMD" --is_h2o --h2o_ratio 0.2"
  fi

  python $UVM_PATH/transformer.py $CMD
  # Remove the shared library after each run to keep the directory clean
  rm allocate.so
done

# ─── SECTION 2: FLEXGEN PATH ─────────────────────────────────────────────────
# Tests four FlexGen-based schemes. FlexGen explicitly offloads weights and KV
# caches across GPU VRAM, CPU DRAM, and disk — unlike UVM which does it
# transparently. Each scheme swaps in a different flex_opt.py and
# pytorch_backend.py via symlinks.

FLEXGEN_PATH=$PWD/../../flexgen

for SCHEME in "original" "int4" "h2o" "infinigen"
do
  # flexgen/flexgen/flex_opt.py is the active entry point used by
  # "python -m flexgen.flex_opt". Swapping the symlink changes which algorithm
  # runs without touching any Python import paths.
  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py

  if [ "$SCHEME" = "int4" ]
  then
    # int4 reuses the original (baseline) code — the only difference is the
    # --compress-cache flag below, which enables 4-bit group quantization on
    # the KV cache, halving its memory footprint.
    ln -s ../original/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
    ln -s ../original/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  else
    ln -s ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
    ln -s ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  fi

  # --percent 100 0 0 100 100 0 means:
  #   weights:     100% GPU,   0% CPU,   0% disk
  #   KV cache:      0% GPU, 100% CPU,   0% disk
  #   activations: 100% GPU,   0% CPU,   0% disk
  # So weights stay on GPU but the KV cache is offloaded to CPU RAM.
  # This is the key offloading scenario FlexGen is designed for.
  CMD="--model huggingface/opt-13b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"

  if [ "$SCHEME" = "int4" ]
  then
    # Compress KV cache to 4-bit integers to reduce CPU memory and PCIe traffic
    CMD=$CMD" --compress-cache"
  elif [ "$SCHEME" = "h2o" ]
  then
    # Keep only 415 KV tokens (~20% of 2048) per decode step.
    # hh-ratio=0.1 means the top 10% are "heavy hitters" always kept.
    # hh-all means apply H2O eviction across all attention layers.
    CMD=$CMD" --max-num-kv 415 --hh-ratio 0.1 --hh-all"
  elif [ "$SCHEME" = "infinigen" ]
  then
    # alpha=4: speculation threshold — tokens within (max_score - 4) are critical.
    # partial-weight-ratio=0.2: use 20% of Q/K columns for cheap speculation.
    # max-num-kv=400: upper bound on KV tokens fetched per decode step (~20% of 2048).
    CMD=$CMD" --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 400"
  fi

  python -m flexgen.flex_opt $CMD
done
