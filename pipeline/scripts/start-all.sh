#!/bin/bash
# Starts the whole pipeline: STT, TTS, LLM (20 expert layers in RAM), bot.
# Reads .env from the repository root if present.
cd "$(dirname "$0")/.."
[ -f ../.env ] && set -a && . ../.env && set +a
mkdir -p logs
bash scripts/start-stt.sh
for i in $(seq 1 60); do grep -q "ready" logs/stt.log 2>/dev/null && break; sleep 3; done
bash scripts/start-tts.sh 0.11 0.05 > /dev/null
for i in $(seq 1 120); do grep -q "Application startup complete" logs/tts.log 2>/dev/null && break; sleep 5; done
bash scripts/start-llm.sh "${N_CPU_MOE:-20}"
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
bash scripts/start-bot.sh
echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
