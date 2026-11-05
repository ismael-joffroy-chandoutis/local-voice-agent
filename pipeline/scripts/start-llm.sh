#!/bin/bash
# Qwen3.6 35B-A3B (Q4_K_M GGUF) served by llama.cpp on 127.0.0.1:8080, thinking disabled.
# Argument: number of MoE expert layers kept in system RAM (--n-cpu-moe). 20 out of 40 was
# the setting measured on a 32 GB card next to the STT and the TTS.
cd "$(dirname "$0")/.." && mkdir -p logs
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
MODEL_GGUF=${MODEL_GGUF:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
N=${1:-20}
setsid nohup "$LLAMA_SERVER" -m "$MODEL_GGUF" -ngl 99 --n-cpu-moe "$N" -c 8192 -np 1 -fa on -t 16 \
  --jinja --reasoning-budget 0 --host 127.0.0.1 --port 8080 > logs/llm.log 2>&1 < /dev/null &
echo $! > logs/llm.pid
