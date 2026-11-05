#!/bin/bash
# Qwen3-TTS 0.6B CustomVoice (default model voices) served by vLLM-Omni on 127.0.0.1:8093,
# with reduced GPU memory so that STT, TTS and LLM share one 32 GB card.
# Arguments: GPU memory share of stage 0 and stage 1 (start-all.sh passes 0.11 and 0.05).
cd "$(dirname "$0")/.." && mkdir -p logs
TTS_PY=${TTS_PY:-python}   # interpreter of the environment where vllm-omni is installed
U0=${1:-0.08}; U1=${2:-0.05}
"$TTS_PY" - "$U0" "$U1" <<'PY'
import sys, yaml, vllm_omni, os
src = os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts.yaml")
c = yaml.safe_load(open(src))
s0, s1 = c["stages"]
s0["gpu_memory_utilization"] = float(sys.argv[1]); s0["max_num_seqs"] = 2; s0["max_model_len"] = 2048
s1["gpu_memory_utilization"] = float(sys.argv[2]); s1["max_num_seqs"] = 2
s1["max_model_len"] = 16384; s1["max_num_batched_tokens"] = 16384
yaml.safe_dump(c, open("qwen3_tts_local.yaml", "w"), sort_keys=False)
print("config written", sys.argv[1:])
PY
VLLM=${VLLM:-vllm}
setsid nohup "$VLLM" serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --deploy-config qwen3_tts_local.yaml --host 127.0.0.1 --port 8093 \
  > logs/tts.log 2>&1 < /dev/null &
