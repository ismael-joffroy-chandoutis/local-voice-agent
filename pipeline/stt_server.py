"""Streaming STT server: Kyutai stt-1b-en_fr (PyTorch, moshi) on one CUDA GPU.

Websocket protocol (ws://127.0.0.1:8792/stt), one connection = one speaking turn:
  client -> bytes: PCM int16 mono 24 kHz, any size (cut into 80 ms frames)
  client -> "FLUSH": end of turn decided by the bot; the server pushes silence at full
                     speed to empty the fixed model delay (0.5 s), then answers
  server -> {"partial": "..."}  word by word
  server -> {"final": "...", "flush_ms": n}  after FLUSH
"""

import asyncio
import json
import math
import time

import moshi.models
import numpy as np
import torch
import websockets

REPO = "kyutai/stt-1b-en_fr"
DEVICE = "cuda"

info = moshi.models.loaders.CheckpointInfo.from_hf_repo(REPO)
mimi = info.get_mimi(device=DEVICE)
tokenizer = info.get_text_tokenizer()
lm = info.get_moshi(device=DEVICE, dtype=torch.bfloat16)
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
FRAME = mimi.frame_size  # 1920 samples at 24 kHz = 80 ms
DELAY_S = info.stt_config.get("audio_delay_seconds", 0.5)
N_FLUSH = math.ceil(DELAY_S * mimi.frame_rate) + 5
SILENCE = torch.zeros((1, 1, FRAME), dtype=torch.float32, device=DEVICE)
LOCK = asyncio.Lock()  # one turn at a time on the GPU
COURANT = None  # connection holding the model; a new connection replaces it


def step(chunk: torch.Tensor) -> str:
    tokens = lm_gen.step(mimi.encode(chunk))
    if tokens is None:
        return ""
    t = tokens[0, 0, 0].item()
    if t in (0, 3):
        return ""
    return tokenizer.id_to_piece(t).replace("▁", " ")


@torch.no_grad()
def warmup():
    with mimi.streaming(1), lm_gen.streaming(1):
        for _ in range(20):
            step(SILENCE)
    torch.cuda.synchronize()


async def handle(ws):
    global COURANT
    if COURANT is not None:
        # an abandoned session (client gone without closing) must not block the next one
        await COURANT.close()
    async with LOCK:
        COURANT = ws
        text, buf = [], np.zeros(0, dtype=np.float32)
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            for _ in range(3):  # short silence prefix, as in the official script
                step(SILENCE)
            async for msg in ws:
                if isinstance(msg, bytes):
                    pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    buf = np.concatenate([buf, pcm])
                    while len(buf) >= FRAME:
                        chunk = torch.from_numpy(buf[:FRAME].copy()).to(DEVICE)[None, None]
                        buf = buf[FRAME:]
                        piece = step(chunk)
                        if piece:
                            text.append(piece)
                            await ws.send(json.dumps({"partial": "".join(text).strip()}))
                elif msg == "FLUSH":
                    t0 = time.perf_counter()
                    if len(buf):
                        pad = np.zeros(FRAME, dtype=np.float32)
                        pad[: len(buf)] = buf
                        text.append(step(torch.from_numpy(pad).to(DEVICE)[None, None]))
                    for _ in range(N_FLUSH):
                        text.append(step(SILENCE))
                    torch.cuda.synchronize()
                    ms = int((time.perf_counter() - t0) * 1000)
                    await ws.send(json.dumps({"final": "".join(text).strip(), "flush_ms": ms}))
                    break


async def main():
    warmup()
    print(f"kyutai stt ready: frame {FRAME}, delay {DELAY_S}s, flush {N_FLUSH} frames", flush=True)
    async with websockets.serve(handle, "127.0.0.1", 8792, max_size=2**24):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
