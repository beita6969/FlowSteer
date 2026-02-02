#!/bin/bash
# vLLM Server Startup Script
# Supports LoRA weight synchronization for P-verl style training

# Activate conda environment (modify path as needed)
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

# Single GPU vLLM inference
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

# Enable vLLM runtime LoRA updating support
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# Model path (modify as needed)
MODEL_PATH="/path/to/Qwen3-8B"

# LoRA weight sync path
LORA_SYNC_PATH="/tmp/vllm_lora_sync"

echo "Starting vLLM server on GPU 0..."
echo "Model: $MODEL_PATH"
echo "LoRA sync path: $LORA_SYNC_PATH"
echo "Runtime LoRA updating: enabled"

# Start vLLM server with LoRA support
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --host 0.0.0.0 \
    --port 8003 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 64
