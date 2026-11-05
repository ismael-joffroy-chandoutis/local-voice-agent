"""Fully local voice bot on Pipecat 1.7 (tested under WSL with one RTX 5090).

Cascade: WebRTC in -> Kyutai stt-1b-en_fr in streaming (stt_server.py, port 8792)
-> end of turn: Silero VAD + Smart Turn v3 + a two-speed guard -> Qwen3.6 35B-A3B with
thinking disabled (llama.cpp server, port 8080) -> Qwen3-TTS 0.6B CustomVoice served by
vLLM-Omni (port 8093) -> WebRTC out. No outbound network call: everything runs on the machine.

Run: scripts/start-bot.sh (listens on 127.0.0.1:7860).
Persona: persona.md, read again at each connection.
At the end of each conversation the bot writes a note (Markdown + JSON) in calls/.

The prompts are in French because the test bench targeted French and English callers;
the persona tells the model to answer in the caller's language.
"""

import json
import os
import re
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import soxr
import websockets
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import ErrorFrame, Frame, InterimTranscriptionFrame, TranscriptionFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.time import time_now_iso8601

STT_WS = os.getenv("STT_WS", "ws://127.0.0.1:8792/stt")
LLM_BASE = os.getenv("LLM_BASE", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b")
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8093/v1/audio/speech")
TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "vivian")
OWNER = os.getenv("OWNER_NAME", "Ismaël")

# Fixed greeting, played at each connection. The persona alone is not enough (on the bench,
# 4 answers out of 15 did not say "AI"), so the disclosure does not depend on the LLM.
GREETING = os.getenv(
    "GREETING",
    f"Bonjour, vous êtes sur l'assistant d'{OWNER}, une intelligence artificielle. Je note votre message et je le lui transmets. Qui est à l'appareil ?",
)
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "standard")  # "standard" (callers) or "dictee" (the owner)
CALLS = Path(os.getenv("CALLS_DIR", str(Path(__file__).with_name("calls"))))

EXTRACTION = f"""Tu reçois la transcription d'une conversation entre un interlocuteur et l'assistant vocal d'{OWNER}.
Remplis la fiche en JSON, avec exactement ces clés :
"mode" : "dictee" si c'est {OWNER} lui-même qui parle (il l'a dit, ou il dicte des notes), sinon "appel" ;
"nom" : nom de l'interlocuteur tel qu'il l'a donné, sinon "" ;
"motif" : motif de l'appel en quelques mots, sinon "" ;
"numero" : numéro de téléphone tel qu'il l'a donné, en chiffres, sinon "" ;
"message" : le message à transmettre à {OWNER}, en une ou deux phrases, sinon "" ;
"taches" : liste des actions à faire (verbe à l'infinitif, échéance si elle est dite), sinon [] ;
"notes" : liste des autres informations dictées par {OWNER} qui ne sont pas des actions, sinon [] ;
un même élément va dans une seule des deux listes ;
"resume" : une phrase qui dit ce qui s'est passé.
N'invente rien : une information absente reste vide."""

# End-of-turn guard, measured on the bench: Smart Turn sure (probability >= FAST_THRESHOLD)
# -> end of turn at once; otherwise GUARD_SILENCE_S of extra silence is required, and any
# speech during the wait cancels the end of turn.
FAST_THRESHOLD = float(os.getenv("FAST_THRESHOLD", "0.985"))
GUARD_SILENCE_S = float(os.getenv("GUARD_SILENCE_S", "0.25"))

FALLBACK = (
    f"Tu es l'assistant d'{OWNER}, une intelligence artificielle, et tu le dis d'emblée. "
    "Réponses orales, une à trois phrases. Aucun engagement en son nom."
)

# Optional access filter. A reverse proxy in front of the bot can add a header carrying the
# authenticated user. When both variables are set, any request whose header holds another
# user gets a 403. Requests without the header (local access) pass.
ALLOWED_USER_HEADER = os.getenv("ALLOWED_USER_HEADER", "")
ALLOWED_USER = os.getenv("ALLOWED_USER", "")


def system_prompt() -> str:
    try:
        text = Path(__file__).with_name("persona.md").read_text(encoding="utf-8").strip().replace("{owner}", OWNER)
    except OSError as e:
        logger.warning(f"persona unreadable ({e}): fallback prompt")
        text = FALLBACK
    if DEFAULT_MODE == "dictee":
        text += f"\n\nRéglage : l'interlocuteur est {OWNER}. Tu es en mode dictée dès le début de l'appel."
    return text


def greeting() -> str:
    if DEFAULT_MODE == "dictee":
        return f"{OWNER}, je t'écoute, je note tout."
    return GREETING


def language(text: str) -> str:
    """French or English, for the TTS language parameter."""
    t = f" {text.lower()} "
    fr = sum(c in t for c in "éèàçêùâîô") + sum(w in t for w in (" je ", " vous ", " est ", " le ", " la ", " les ", " pour ", " et ", " que "))
    en = sum(w in t for w in (" the ", " is ", " you ", " and ", " to ", " i ", " can ", " this ", " for "))
    return "French" if fr >= en else "English"


class KyutaiStreamingSTT(STTService):
    """Streams audio to Kyutai as it arrives; flushes the model delay when Silero sees
    0.2 s of silence and the end-of-turn rule agrees, then emits the final transcript."""

    def __init__(self, **kwargs):
        super().__init__(sample_rate=16000, **kwargs)
        self._ws = None
        self._vad = SileroVADAnalyzer(sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6))
        self._vad.set_sample_rate(16000)
        self._speaking = False
        self._rest = b""
        self._turn = LocalSmartTurnAnalyzerV3()
        self._segment16: list[bytes] = []
        self._waiting: float | None = None  # start of the guard wait

    async def _flush(self) -> str:
        await self._ws.send("FLUSH")
        while True:
            r = json.loads(await asyncio_wait(self._ws.recv(), 3.0))
            if "final" in r:
                break
        await self._ws.close()
        self._ws = None
        return r["final"]

    async def _close(self):
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def stop(self, frame):
        await self._close()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._close()
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            if self._ws is None:
                self._ws = await websockets.connect(STT_WS, max_size=2**24)
            x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
            x24 = (soxr.resample(x, 16000, 24000) * 32767).astype(np.int16)
            await self._ws.send(x24.tobytes())
            # non-blocking partials
            try:
                while True:
                    msg = await asyncio_wait(self._ws.recv(), 0)
                    p = json.loads(msg).get("partial")
                    if p:
                        yield InterimTranscriptionFrame(text=p, user_id=self._user_id, timestamp=time_now_iso8601(), language=Language.FR)
            except TimeoutError:
                pass
            self._rest += audio
            self._segment16.append(audio)
            self._segment16 = self._segment16[-400:]  # 8 s max, what Smart Turn reads
            state = None
            while len(self._rest) >= 1024:
                state = await self._vad.analyze_audio(self._rest[:1024])
                self._rest = self._rest[1024:]
                if state == VADState.SPEAKING:
                    self._speaking = True
            end = False
            if self._waiting is not None:
                if state in (VADState.STARTING, VADState.SPEAKING) or self._speaking:
                    logger.info("guard: speech resumed, end of turn cancelled")
                    self._waiting = None
                elif time.time() - self._waiting >= GUARD_SILENCE_S:
                    logger.info("guard: silence confirmed, end of turn")
                    self._waiting = None
                    end = True
            if self._speaking and state == VADState.QUIET:
                self._speaking = False
                x16 = np.frombuffer(b"".join(self._segment16), dtype=np.int16).astype(np.float32) / 32768
                turn = self._turn._predict_endpoint(x16)
                p = float(turn.get("probability", 1))
                logger.info(f"smart turn probability {p:.3f} prediction {turn.get('prediction')}")
                if turn.get("prediction", 1) == 1:
                    if p >= FAST_THRESHOLD:
                        end = True
                    else:
                        self._waiting = time.time()
            if end:
                t0 = time.time()
                text = await self._flush()
                self._segment16 = []
                logger.info(f"LATENCY stt flush {int((time.time() - t0) * 1000)}ms -> {text!r}")
                if text:
                    yield TranscriptionFrame(text=text, user_id=self._user_id, timestamp=time_now_iso8601(), language=Language.FR, finalized=True)
        except Exception as e:
            self._ws = None
            yield ErrorFrame(f"Kyutai STT: {e}")


async def asyncio_wait(coro, timeout):
    import asyncio

    return await asyncio.wait_for(coro, timeout=timeout if timeout > 0 else 0.001)


class QwenOmniTTS(TTSService):
    """Sentence -> vLLM-Omni /v1/audio/speech, streamed as 24 kHz PCM."""

    def __init__(self, **kwargs):
        super().__init__(sample_rate=24000, **kwargs)
        self._client = httpx.AsyncClient(timeout=120)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        try:
            await self.start_tts_usage_metrics(text)
            t0 = time.time()
            first = None
            body = {"input": text, "voice": TTS_VOICE, "language": language(text),
                    "stream": True, "stream_format": "audio", "response_format": "pcm"}
            rest = b""
            async with self._client.stream("POST", TTS_URL, json=body) as r:
                async for chunk in r.aiter_bytes():
                    chunk = rest + chunk
                    n = len(chunk) - len(chunk) % 2
                    rest = chunk[n:]
                    if not n:
                        continue
                    if first is None:
                        first = int((time.time() - t0) * 1000)
                        await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(audio=chunk[:n], sample_rate=24000, num_channels=1, context_id=context_id)
            logger.info(f"LATENCY tts first-audio {first}ms {text[:50]!r}")
        except Exception as e:
            yield ErrorFrame(f"Qwen3-TTS: {e}")


async def run_bot(transport: BaseTransport):
    stt = KyutaiStreamingSTT()
    tts = QwenOmniTTS()
    llm = OpenAILLMService(
        api_key="local",
        base_url=LLM_BASE,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
    start = datetime.now()
    context = LLMContext(messages=[{"role": "system", "content": system_prompt()}, {"role": "assistant", "content": greeting()}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6)),
        ),
    )
    pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frames([TTSSpeakFrame(greeting())])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        await write_note(start, context)


async def write_note(start: datetime, context: LLMContext):
    """At the end of the call: full transcript + note filled by the local LLM, in calls/."""
    turns = []
    for m in context.get_messages():
        role, content = m.get("role"), m.get("content")
        if role == "system" or not content:
            continue
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        turns.append(("Caller" if role == "user" else "Assistant", str(content).strip()))
    if not any(r == "Caller" for r, _ in turns):
        logger.info("call without speech from the caller: no note")
        return
    end = datetime.now()
    text = "\n".join(f"{r}: {c}" for r, c in turns)
    note = {}
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{LLM_BASE}/chat/completions", json={
                "model": LLM_MODEL, "temperature": 0, "max_tokens": 600,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": EXTRACTION}, {"role": "user", "content": text}]})
            note = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning(f"note: extraction failed ({e}), transcript only")
    note.update({"debut": start.isoformat(timespec="seconds"), "fin": end.isoformat(timespec="seconds"),
                 "duree_s": int((end - start).total_seconds())})
    name = note.get("nom") or ("owner" if note.get("mode") == "dictee" else "unknown")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name.encode("ascii", "ignore").decode()).strip("-")[:30] or "unknown"
    base = CALLS / f"{start:%Y-%m-%d-%H%M%S}-{slug}"
    CALLS.mkdir(parents=True, exist_ok=True)

    def line(key, label):
        v = note.get(key)
        return f"- **{label}**: {v}" if v else f"- **{label}**: (not given)"

    items = lambda key: "\n".join(f"- {x}" for x in note.get(key) or []) or "(none)"
    md = "\n".join([
        f"# {'Dictation' if note.get('mode') == 'dictee' else 'Message from ' + (note.get('nom') or 'unknown')}",
        "",
        f"{start:%Y-%m-%d %H:%M} · {note['duree_s']} s · mode {note.get('mode', '?')}",
        "",
        "## Note", "",
        line("nom", "Name"), line("motif", "Reason"), line("numero", "Number"), line("message", "Message"),
        f"- **Summary**: {note.get('resume', '')}",
        "", "## Notes", "", items("notes"),
        "", "## Tasks", "", items("taches"),
        "", "## Transcript", "",
        *[f"**{r}**: {c}  " for r, c in turns],
        "",
    ])
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    base.with_suffix(".json").write_text(json.dumps({**note, "transcription": turns}, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"note written: {base}.md")


async def bot(runner_args: RunnerArguments):
    transport = SmallWebRTCTransport(
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        webrtc_connection=runner_args.webrtc_connection,
    )
    await run_bot(transport)


def warm_up():
    """The first vLLM-Omni request costs about 3 s: pay it at startup, not on the first call."""
    try:
        t0 = time.time()
        with httpx.Client(timeout=60) as c:
            r = c.post(TTS_URL, json={"input": "Bonjour.", "voice": TTS_VOICE, "language": "French", "response_format": "pcm"})
        logger.info(f"TTS warm-up {r.status_code} in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"TTS warm-up failed: {e}")
    # llama.cpp keeps the prefix of the last request in cache: let it read the persona once
    try:
        t0 = time.time()
        with httpx.Client(timeout=60) as c:
            r = c.post(f"{LLM_BASE}/chat/completions", json={
                "model": LLM_MODEL, "max_tokens": 1,
                "messages": [{"role": "system", "content": system_prompt()}, {"role": "assistant", "content": greeting()}, {"role": "user", "content": "Bonjour."}],
                "chat_template_kwargs": {"enable_thinking": False}})
        logger.info(f"LLM warm-up {r.status_code} in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"LLM warm-up failed: {e}")


def access_filter():
    """When enabled, refuses every request whose identity header is missing or names another user.
    The reverse proxy must set this header itself and drop any copy sent by the client."""
    if not (ALLOWED_USER_HEADER and ALLOWED_USER):
        return
    from pipecat.runner import run as runner
    from starlette.responses import PlainTextResponse

    async def check(request, call_next):
        who = request.headers.get(ALLOWED_USER_HEADER.lower())
        if not who or who.strip().lower() != ALLOWED_USER.lower():
            logger.warning(f"access refused to {who or 'request without identity header'}")
            return PlainTextResponse("access restricted", status_code=403)
        return await call_next(request)

    runner.app.middleware("http")(check)  # the runner app is created at import time


if __name__ == "__main__":
    from pipecat.runner.run import main

    access_filter()
    warm_up()
    main()
