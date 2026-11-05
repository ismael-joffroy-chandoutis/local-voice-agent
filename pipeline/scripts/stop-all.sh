#!/bin/bash
# Stops the whole local voice pipeline and frees the GPU
pkill -f "bot.py --host 127.0.0.1 --port 7860"
pkill -f "vllm serve"; pkill -f StageEngineCore
pkill -f stt_server.py
pkill -f llama-server
sleep 8
echo "VRAM after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
