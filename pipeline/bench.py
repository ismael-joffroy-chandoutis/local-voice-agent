"""End-to-end bench of the local voice pipeline, from audio files (no microphone).

A recorded question is played in real time through the chain:
  Silero VAD (0.2 s stop) -> Smart Turn v3 (+ two-speed guard) -> Kyutai STT (flush)
  -> Qwen3.6 35B-A3B (llama.cpp, thinking disabled) -> Qwen3-TTS served by vLLM-Omni
Measured latency = end of the user's speech -> first sample of the answer.
Output: one WAV per exchange (question, silence equal to the measured latency, answer) + JSON.

Usage: python bench.py --out DIR [--tours 3] [--sans-garde]
Questions are WAV files made once by make_questions.py, in questions/.
"""

import argparse
import os
import asyncio
import json
import re
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import soxr
import websockets
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

STT_WS = "ws://127.0.0.1:8792/stt"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
QWEN_TTS_URL = "http://127.0.0.1:8093/v1/audio/speech"
SR = 24000
FRAME = 1920  # 80 ms

OWNER = os.getenv("OWNER_NAME", "Ismaël")
SYSTEME = Path(__file__).with_name("persona.md").read_text(encoding="utf-8").replace("{owner}", OWNER)
# Two-speed end-of-turn guard: Smart Turn very sure (probability >= SEUIL_RAPIDE) -> end of
# turn at once; otherwise GARDE_SILENCE_S of extra silence is required (the pause between two
# sentences lasts 0.3 to 0.4 s and ends before), and resumed speech cancels the end of turn.
SEUIL_RAPIDE = 0.985
GARDE_SILENCE_S = 0.25
GARDE = True

QUESTIONS = [  # WAV files made once with Kokoro (make_questions.py), folder questions/
    ("fr1", "fr", "ff_siwis", "Bonjour, tu peux me rappeler pourquoi on garde une marge de quinze pour cent sur les disques ?"),
    ("fr2", "fr", "ff_siwis", "Donne-moi une idée simple pour le dîner de ce soir, avec ce qu'on trouve dans un placard."),
    ("en1", "en", "af_heart", "Hi, can you explain in two sentences what a voice activity detector does?"),
    ("fr3", "fr", "ff_siwis", "Bonjour, je suis Claire Martin, je voudrais parler à Ismaël pour un projet de film."),
    ("en2", "en", "af_heart", "Hello, this is John from the festival. Can I speak to Ismaël?"),
]
QWEN_VOIX = {"fr": ("vivian", "French"), "en": ("ryan", "English")}


async def tts_stream(engine: str, text: str, lang: str, client: httpx.AsyncClient):
    """Generator of PCM int16 24 kHz chunks."""
    voice, language = QWEN_VOIX[lang]
    body = {
        "input": text, "voice": voice, "language": language,
        "stream": True, "stream_format": "audio", "response_format": "pcm",
    }
    async with client.stream("POST", QWEN_TTS_URL, json=body, timeout=120) as r:
        r.raise_for_status()
        reste = b""
        async for chunk in r.aiter_bytes():
            chunk = reste + chunk
            n = len(chunk) - len(chunk) % 2
            reste = chunk[n:]
            if n:
                yield chunk[:n]


def derniere_parole(pcm: np.ndarray) -> int:
    """Index of the last speech sample (energy over 10 ms frames)."""
    f = 240
    e = np.array([np.abs(pcm[i:i + f]).mean() for i in range(0, len(pcm) - f, f)])
    seuil = max(e.max() * 0.05, 1e-4)
    idx = np.where(e > seuil)[0]
    return int((idx[-1] + 1) * f) if len(idx) else len(pcm)


async def echange(nom, lang, voix_q, question, engine, out: Path, client):
    with wave.open(str(Path(__file__).with_name("questions") / f"{nom}.wav")) as w:
        q_pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    fin = derniere_parole(q_pcm.astype(np.float32) / 32768)
    q_pcm = q_pcm[: fin + int(0.05 * SR)]
    flux = np.concatenate([q_pcm, np.zeros(int(3.0 * SR), dtype=np.int16)])

    vad = SileroVADAnalyzer(sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6))
    vad.set_sample_rate(16000)
    turn = LocalSmartTurnAnalyzerV3()
    m = {"nom": nom, "tts": engine, "question": question}
    entendu16 = []
    reste16 = np.zeros(0, dtype=np.int16)
    parle = False
    t_parole0 = None
    attente = None

    async with websockets.connect(STT_WS, max_size=2**24) as stt:
        t_debut = time.perf_counter()
        t_eos = None
        for i in range(0, len(flux), FRAME):
            cible = t_debut + i / SR
            await asyncio.sleep(max(0, cible - time.perf_counter()))
            bloc = flux[i:i + FRAME]
            if t_eos is None and i + FRAME >= fin:
                t_eos = t_debut + fin / SR
            await stt.send(bloc.tobytes())
            b16 = soxr.resample(bloc.astype(np.float32) / 32768, SR, 16000)
            b16 = (b16 * 32767).astype(np.int16)
            entendu16.append(b16)
            reste16 = np.concatenate([reste16, b16])
            etat = None
            while len(reste16) >= 512:
                etat = await vad.analyze_audio(reste16[:512].tobytes())
                reste16 = reste16[512:]
                if etat == VADState.SPEAKING:
                    parle = True
            if etat == VADState.SPEAKING and t_parole0 is None:
                t_parole0 = time.perf_counter()
            if attente is not None:
                # guard active: wait for resumed speech or for GARDE_SILENCE_S of silence
                if etat in (VADState.STARTING, VADState.SPEAKING) or parle:
                    m.setdefault("garde", []).append("reprise de parole, fin annulée")
                    attente = None
                elif time.perf_counter() - attente[0] >= GARDE_SILENCE_S:
                    t_vad, tour, t_turn = attente
                    complet = True
                    m.setdefault("garde", []).append("fin acceptée après silence long")
                    attente = None
                    if t_eos is not None:
                        m["vad_ms"] = int((t_vad - t_eos) * 1000)
                        m["smart_turn_ms"] = int((t_turn - t_vad) * 1000)
                        m["smart_turn"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in tour.items()}
                        m["attente_garde_ms"] = int((time.perf_counter() - t_turn) * 1000)
                        break
                    m.setdefault("coupures_prematurees", []).append(round(time.perf_counter() - t_debut, 2))
                    break
            if parle and etat == VADState.QUIET:
                t_vad = time.perf_counter()
                tour = turn._predict_endpoint(np.concatenate(entendu16).astype(np.float32) / 32768)
                t_turn = time.perf_counter()
                complet = tour.get("prediction", 1) == 1
                if GARDE and complet and tour.get("probability", 1) < SEUIL_RAPIDE:
                    attente = (t_vad, tour, t_turn)
                    m.setdefault("garde", []).append(f"attente à {t_vad - t_debut:.2f}s, proba {tour.get('probability', -1):.3f}")
                    parle = False
                    continue
                if t_eos is None:
                    # pause inside the sentence: Smart Turn should say "incomplete"
                    m.setdefault("pauses_internes", []).append({"complet": bool(complet), "proba": float(tour.get("probability", -1))})
                    if complet:
                        m.setdefault("coupures_prematurees", []).append(round(t_vad - t_debut, 2))
                    parle = False
                    continue
                m["vad_ms"] = int((t_vad - t_eos) * 1000)
                m["smart_turn_ms"] = int((t_turn - t_vad) * 1000)
                m["smart_turn"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in tour.items()}
                if complet:
                    break
                parle = False  # turn judged incomplete: keep listening
        await stt.send("FLUSH")
        while True:
            r = json.loads(await stt.recv())
            if "final" in r:
                break
    t_stt = time.perf_counter()
    m["stt_flush_ms"] = r["flush_ms"]
    m["transcription"] = r["final"]

    # streaming LLM, thinking disabled; cut at the first strong punctuation mark
    corps = {
        "messages": [{"role": "system", "content": SYSTEME}, {"role": "user", "content": r["final"]}],
        "stream": True, "temperature": 0.7, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    texte, segments, t_tok1 = "", [], None
    reponse_pcm = bytearray()
    t_audio1 = None
    file_tts: asyncio.Queue = asyncio.Queue()

    async def lecteur():
        nonlocal t_audio1
        while (seg := await file_tts.get()) is not None:
            async for morceau in tts_stream(engine, seg, lang, client):
                if t_audio1 is None:
                    t_audio1 = time.perf_counter()
                reponse_pcm.extend(morceau)

    tache = asyncio.create_task(lecteur())
    async with client.stream("POST", LLM_URL, json=corps, timeout=120) as resp:
        async for ligne in resp.aiter_lines():
            if not ligne.startswith("data: ") or ligne.endswith("[DONE]"):
                continue
            d = json.loads(ligne[6:])["choices"][0]["delta"].get("content") or ""
            if d and t_tok1 is None:
                t_tok1 = time.perf_counter()
            texte += d
            coupe = re.search(r"^(.+?[.!?…])(\s|$)", texte, re.S)
            if not segments and not coupe:
                coupe = re.search(r"^(.{25,}?,)\s", texte, re.S)  # shorter first segment
            if coupe:
                seg = coupe.group(1).strip()
                if not segments:
                    m["llm_premier_segment_ms"] = int((time.perf_counter() - t_stt) * 1000)
                segments.append(seg)
                await file_tts.put(seg)
                texte = texte[coupe.end():]
    if texte.strip():
        segments.append(texte.strip())
        await file_tts.put(texte.strip())
    t_llm_fin = time.perf_counter()
    await file_tts.put(None)
    await tache

    m["llm_ttft_ms"] = int((t_tok1 - t_stt) * 1000)
    m["llm_total_ms"] = int((t_llm_fin - t_stt) * 1000)
    m["reponse"] = " ".join(segments)
    m["tts_premier_son_ms"] = int((t_audio1 - t_stt) * 1000) - m.get("llm_premier_segment_ms", 0)
    m["latence_bout_en_bout_ms"] = int((t_audio1 - t_eos) * 1000)

    blanc = np.zeros(int(m["latence_bout_en_bout_ms"] / 1000 * SR), dtype=np.int16)
    tout = np.concatenate([q_pcm, blanc, np.frombuffer(bytes(reponse_pcm), dtype=np.int16)])
    with wave.open(str(out / f"echange-{nom}-{engine}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(tout.tobytes())
    return m


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default="qwen", choices=["qwen"])
    ap.add_argument("--out", default=".")
    ap.add_argument("--sans-garde", action="store_true")
    ap.add_argument("--tours", type=int, default=1, help="repetitions per question (median latency)")
    a = ap.parse_args()
    global GARDE
    GARDE = not a.sans_garde
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    async with httpx.AsyncClient() as client:
        for nom, lang, vq, q in QUESTIONS:
            for k in range(a.tours):
                m = await echange(nom, lang, vq, q, a.tts, out, client)
                m["tour"] = k
                print(json.dumps(m, ensure_ascii=False), flush=True)
                res.append(m)
    (out / f"mesures-{a.tts}{'-sans-garde' if a.sans_garde else ''}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
