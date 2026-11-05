"""Real test of the Pipecat bot over WebRTC (aiortc), without a microphone: it "speaks" a WAV.

It goes through the same door as a phone browser: POST <BOT_URL>/api/offer.
Measures: end of the question -> first sound of the answer received (energy of the incoming stream).
Usage: BOT_URL=https://your-host python test_webrtc.py questions/fr3.wav out.wav [initial_silence_s]
"""

import asyncio
import fractions
import os
import sys
import time
import wave

import av
import httpx
import numpy as np
import soxr
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

URL = os.getenv("BOT_URL", "http://127.0.0.1:7860").rstrip("/") + "/api/offer"
SR = 48000
F = 960  # 20 ms


class PisteWav(MediaStreamTrack):
    kind = "audio"

    def __init__(self, pcm48: np.ndarray):
        super().__init__()
        self.pcm = pcm48
        self.i = 0
        self.t0 = None
        self.t_fin_parole = None
        self.fin_index = None

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        n = self.i // F
        await asyncio.sleep(max(0, self.t0 + n * F / SR - time.perf_counter()))
        bloc = self.pcm[self.i:self.i + F]
        if len(bloc) < F:
            bloc = np.concatenate([bloc, np.zeros(F - len(bloc), dtype=np.int16)])
        if self.fin_index is not None and self.t_fin_parole is None and self.i + F >= self.fin_index:
            self.t_fin_parole = self.t0 + self.fin_index / SR
        self.i += F
        fr = av.AudioFrame.from_ndarray(bloc.reshape(1, -1), format="s16", layout="mono")
        fr.sample_rate = SR
        fr.pts = self.i
        fr.time_base = fractions.Fraction(1, SR)
        return fr


async def main(entree, sortie):
    with wave.open(entree) as w:
        sr = w.getframerate()
        q = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    q48 = soxr.resample(q, sr, SR)
    e = np.array([np.abs(q48[i:i + 480]).mean() for i in range(0, len(q48) - 480, 480)])
    fin = int((np.where(e > e.max() * 0.05)[0][-1] + 1) * 480)
    debut = int(float(sys.argv[3]) * SR) if len(sys.argv) > 3 else int(2.0 * SR)  # initial silence (greeting)
    pcm = np.concatenate([np.zeros(debut), q48[:fin], np.zeros(12 * SR)])
    piste = PisteWav((pcm * 32767).astype(np.int16))
    piste.fin_index = debut + fin

    pc = RTCPeerConnection()
    recu, t_premier_son = [], None
    fini = asyncio.Event()

    @pc.on("track")
    def on_track(track):
        async def lire():
            nonlocal t_premier_son
            while True:
                try:
                    fr = await track.recv()
                except Exception:
                    break
                x = fr.to_ndarray().astype(np.float32).reshape(-1)
                if fr.layout.name == "stereo":
                    x = x[::2]
                x = x / 32768
                recu.append((time.perf_counter(), fr.sample_rate, x))
                if piste.t_fin_parole and t_premier_son is None and np.abs(x).mean() > 0.003:
                    t_premier_son = time.perf_counter()
                if piste.i >= len(piste.pcm):
                    fini.set()
                    break
        asyncio.ensure_future(lire())

    pc.addTrack(piste)
    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(URL, json={"sdp": pc.localDescription.sdp, "type": "offer"})
        print("offer", r.status_code)
        ans = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    try:
        await asyncio.wait_for(fini.wait(), timeout=len(pcm) / SR + 15)
    except asyncio.TimeoutError:
        pass
    await pc.close()

    if t_premier_son and piste.t_fin_parole:
        print(f"LATENCE_WEBRTC {int((t_premier_son - piste.t_fin_parole) * 1000)} ms")
    else:
        print("no answer detected", piste.t_fin_parole, t_premier_son)
    if recu:
        sr_out = recu[0][1]
        audio = np.concatenate([x for _, _, x in recu])
        with wave.open(sortie, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr_out)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print("received", round(len(audio) / sr_out, 1), "s at", sr_out, "Hz")


asyncio.run(main(sys.argv[1], sys.argv[2]))
