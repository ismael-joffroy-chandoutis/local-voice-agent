#!/bin/bash
# Pipecat bot on 127.0.0.1:7860 (WebRTC). Put an HTTPS reverse proxy in front of it:
# browsers only give the microphone to a secure page.
cd "$(dirname "$0")/.." && mkdir -p logs
BOT_PY=${BOT_PY:-python}   # interpreter of the environment where pipecat-ai is installed
pkill -f "bot.py --host 127.0.0.1 --port 7860"; sleep 1
setsid nohup "$BOT_PY" bot.py --host 127.0.0.1 --port 7860 -t webrtc > logs/bot.log 2>&1 < /dev/null &
echo $! > logs/bot.pid
