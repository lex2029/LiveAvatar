#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

export ENABLE_COMPILE="${ENABLE_COMPILE:-false}"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "720*1280" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model True \
     --convert_model_dtype \
     --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge."  \
     --image "examples/dwarven_blacksmith.jpg" \
     --audio "examples/dwarven_blacksmith.wav" \
     --infer_frames 48 \
     --load_lora \
     --lora_path_dmd "ckpt/LiveAvatar/liveavatar.safetensors" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 10000 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8
     
