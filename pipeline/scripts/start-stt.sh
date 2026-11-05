#!/bin/bash
# Kyutai stt-1b-en_fr streaming server on 127.0.0.1:8792
cd "$(dirname "$0")/.." && mkdir -p logs
STT_PY=${STT_PY:-python}   # interpreter of the environment where moshi is installed
setsid nohup "$STT_PY" -u stt_server.py > logs/stt.log 2>&1 < /dev/null &
echo $! > logs/stt.pid
