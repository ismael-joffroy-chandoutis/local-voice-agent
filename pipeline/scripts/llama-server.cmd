@echo off
rem Windows variant of start-llm.sh (the measured setup ran llama.cpp natively on Windows
rem and the other components in WSL). Usage: llama-server.cmd 20
llama-server.exe -m "%MODEL_GGUF%" -ngl 99 --n-cpu-moe %1 -c 8192 -np 1 -fa on -t 16 --jinja --reasoning-budget 0 --host 127.0.0.1 --port 8080 > llm.log 2>&1
