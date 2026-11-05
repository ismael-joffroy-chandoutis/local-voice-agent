<picture><source media="(prefers-color-scheme: dark)" srcset="banner.svg"><img src="banner-light.svg" alt="Local voice agent" width="100%"></picture>

[English](README.md) · [Français](README.fr.md)

# Local voice agent

Ismaël Joffroy Chandoutis is an artist and filmmaker who runs Claude Code as the operating layer of his studio. In November 2026 he had a fully local voice assistant built on a single RTX 5090: it answered in a browser, in French and English, took messages for him and wrote a note at the end of each conversation. On 5 November 2026 he spoke to it himself for 57 seconds. The transcripts were accurate, no sentence was cut off, and the answers came back in 1.0 to 1.4 s on the server side. The role was wrong: the assistant was set up to take messages from third parties and had nothing to do for its owner. He wanted an agent that phones services on his behalf to book, move an appointment, ask for a price or follow up on a file. The local pipeline was paused as it stood and the calling agent was designed next, on ElevenLabs with Claude as the reasoning model, behind a written mission order and a human go.

<img src="monde.jpg" alt="world image" width="100%">

This repository holds both parts. `pipeline/` is the local pipeline, with the code that ran and the measurements it produced. `caller/` documents the calling agent: its architecture, its mission order template and its go gate. No code for the calling agent has run yet, so none is published.

## Part 1: a voice pipeline that runs on one GPU

### Architecture

```
browser (WebRTC) ──> Pipecat bot (bot.py, port 7860)
                       │
                       ├─ Silero VAD, 0.2 s of silence
                       ├─ Smart Turn v3 + two-speed guard (end of turn)
                       ├─ Kyutai stt-1b-en_fr, streaming (stt_server.py, port 8792)
                       ├─ Qwen3.6 35B-A3B, Q4_K_M GGUF, thinking disabled (llama.cpp, port 8080)
                       └─ Qwen3-TTS 0.6B CustomVoice served by vLLM-Omni (port 8093)
                       │
                       └─ end of call: the local LLM fills a note (JSON + Markdown) in calls/
```

Every component listens on 127.0.0.1 and no request leaves the machine. In the measured setup, llama.cpp ran natively on Windows and the other components ran in WSL on the same machine. The TTS used a default voice of the model (`vivian` in French, `ryan` in English on the bench). No voice was cloned.

Three rules come from the measurements:

1. A fixed greeting is played at each connection and says the assistant is an AI. With the persona alone, 11 answers out of 15 on the bench said so. The four others were the English technical question (3 runs out of 3) and one run of a French question.
2. The end of turn uses two speeds. If Smart Turn gives a probability of at least 0.985, the turn ends at once. Below that, 0.25 s of extra silence is required, and any speech during the wait cancels the end of turn. The pauses inside the test questions lasted 0.28 to 0.36 s, and Smart Turn judged three of them finished, with probabilities of 0.855, 0.965 and 0.977. Real ends of turn scored 0.938 to 0.991. The threshold comes from five synthetic questions only and should be tuned on real voices (`FAST_THRESHOLD`, `GUARD_SILENCE_S`).
3. 20 of the 40 MoE expert layers of the LLM stay in system RAM (`--n-cpu-moe 20`). The full Q4 model needs about 22.5 GB on the card, and about 14.3 GB remained once the STT and the TTS were loaded.

The persona (`pipeline/persona.md`) has two modes. In call mode the assistant says it is an AI, takes no commitment, collects the caller's name, reason, number and message, and reads them back. In dictation mode, triggered when the owner names himself, it switches to short acknowledgements and sorts what he says into notes and tasks. The prompts are written in French because the bench targeted French and English callers.

### Measurements (5 November 2026)

All figures were measured on the same machine on 5 November 2026. Raw records are in `pipeline/measurements/`.

Test bench, 15 exchanges (5 synthetic questions, 3 runs each), latency from the end of the question to the first sound of the answer, server side:

| Variant | Cut-offs in mid-sentence | Median latency | Range |
|---|---|---|---|
| Two-speed guard | 0 out of 15 | 0.88 s | 0.57 to 1.17 s |
| No guard | 9 out of 15 | about 0.65 s (14 runs, cold first call excluded) | |

Median per step, with the guard:

| Step | Median |
|---|---|
| Silero sees 0.2 s of silence | 182 ms |
| Smart Turn v3 | 32 ms |
| Guard wait, when it applies (9 turns out of 15) | 289 ms |
| STT flush | 126 ms |
| LLM to the first sentence | 215 ms |
| of which first token | 54 ms |
| TTS first sound | 32 ms |

Real Pipecat bot, full WebRTC sessions through an HTTPS address, with `test_webrtc.py` playing a WAV as the microphone:

| Measure | Range | Median |
|---|---|---|
| Server side, end of speech to "Bot started speaking" (11 turns) | 0.76 to 1.44 s | 1.13 s |
| Client side, end of the question to the first sound received (12 turns) | 1.85 to 2.84 s | 2.2 s |

The bot adds about 0.25 s to the bench: it waits for a full sentence before the TTS, and the first turn of a session has a slower first token (0.5 to 0.9 s, against 0.2 s afterwards). The gap of about 1 s between server and client mixes network, Opus encoding and the receive buffer of the test client; the measurements do not separate these parts. A phone browser has its own buffers that were not measured.

GPU memory (`nvidia-smi`, MiB): 9,047 before loading (other workloads of the machine and the display), 12,258 with the STT, 18,305 with the STT and the TTS, 30,719 with the LLM (12.4 GB for the LLM with 20 expert layers in RAM), 30,843 at the peak of the bench, on a card of 32.6 GB. LLM throughput rose from 49 to 61 tokens per second with 30 layers in RAM to 63 to 81 with 20.

### Running it again

Requirements: one CUDA GPU with 32 GB, Linux or WSL, a llama.cpp build with `llama-server`, the GGUF file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. The models `kyutai/stt-1b-en_fr` and `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` are downloaded from Hugging Face on first start. The package versions that were installed are listed in `pipeline/requirements-*.txt`; the measured setup used three separate Python environments (STT, TTS, bot).

```
cp .env.example .env              # set MODEL_GGUF and the interpreters of each environment
bash pipeline/scripts/start-all.sh
bash pipeline/scripts/stop-all.sh # stops everything and prints the GPU memory left in use
```

Put an HTTPS reverse proxy in front of port 7860: browsers give the microphone only to a secure page. If the proxy adds a header with the authenticated user, `ALLOWED_USER_HEADER` and `ALLOWED_USER` make the bot refuse every other user with a 403.

To rerun the bench, generate the questions once (`python pipeline/make_questions.py`, Kokoro on CPU), then `python pipeline/bench.py --out results --tours 3`. To test the real bot without a microphone: `BOT_URL=https://your-host python pipeline/test_webrtc.py questions/fr3.wav answer.wav`. `make_scenarios.py` builds two multi-turn scenarios, a caller leaving a message and a dictation. The names and the phone number in the test questions are fictional. Nothing restarts by itself after a reboot.

### Limits

- The bench used synthetic voices. The guard threshold and the latency felt on a phone remain to be checked with real voices.
- Each bench exchange is a single turn. Reading back the name, reason and number over several turns was tested only in two simulated sessions, where the assistant skipped reading the number digit by digit.
- The STT heard the company name "Lumen Films" as "L'Humaine Film", and the note copied the error. "Ismaël" was sometimes heard as "Ismail" in English.
- The last sentence of the assistant, cut by the disconnection, is only partly in the transcript.
- A local model of this size answers and takes notes. Whether it can hold a negotiation or a decision bound by rules was not tested; the calling agent below uses Claude for that reason.
- No telephone line is connected. Calls come from a browser over WebRTC.

## Part 2: an agent that calls, behind a go gate

Status on 5 November 2026: design stage. No call has been placed and no agent has been configured for it. This part describes what the design sets out, and the template in `caller/` is a text document.

### Architecture

```
owner ── mission order written with Claude Code ── "go envoi <name>"
                          │
                          ▼
   ElevenLabs outbound call API + variables of the mission
                          │
   telephony (a number dedicated to the assistant, or SIP) ── call ── the service
                          │
   ElevenLabs agent: ElevenLabs STT → Claude → voice
       tools: calendar (read availability, create an event), keypad tones,
              voicemail detection, transfer to the owner, flag for review, end call
                          │
   end of call → webhook: transcript + extracted fields
                          │
   report (Claude Code): result, exact quote of the confirmation, next steps
```

Design choices recorded in the source notes:

- One generic agent. Each call receives its mission through dynamic variables and per-call overrides (instructions, first message, a strict list of allowed tools), so nothing carries over from one call to the next.
- A text model behind the voice. Claude Opus 5.5 for missions that require a decision, Claude Sonnet 5.5 for simple ones, each falling back to the other, with low reasoning effort during the conversation. The τ-Voice study (Sierra and Princeton, March 2026) reports 31 to 51 % task success on rule-bound phone tasks for speech-to-speech agents, against 85 % for a text model that reasons. The latency of Opus 5.5 on the phone has not been measured.
- Calendar: read availability and create events only. The update and delete tools, which ElevenLabs itself flags as privileged, stay disabled.
- After the call: structured data collection (result, date obtained, price stated, exact quote of the confirmation, next steps) and the transcript, so that the report can be checked against the call.

### The go gate

Before the call:

1. The owner states what he wants in one sentence.
2. Claude writes the mission order from the template, checks the number at its source (official site, earlier email) and reads the calendar for slots. The number to call always comes from the mission order and never from the agent.
3. The full mission order is displayed. Any change means displaying it again.
4. The owner answers `go envoi <mission name>`, alone in the message. A go inside a message that carries other instructions arms nothing. A go is valid for one call, within 30 minutes.

During the call:

5. The first message says, every time, that this is the owner's AI assistant with a synthetic voice, why it calls, and that the call is recorded. It is repeated to each new human after a voice menu or a transfer.
6. Anything the mission order does not allow gets "I need to check with him, he will get back to you", or a transfer to the owner if the order permits it.
7. Before hanging up, the agent sums up aloud and asks for confirmation.

After the call:

8. A report within five minutes: result (obtained, partial, failed), what was agreed word for word, price, date, next actions, gaps from the order, links to the recording and the transcript.
9. A confirmed appointment becomes a calendar event right away, with the context in its notes.
10. A callback or a follow-up is a new mission order with a new go.

Hard rules in every mission: no payment, no bank details, no code received by text message or email, no password, no contract or subscription, no claim to be the owner, and "yes" to the question "are you an AI?". For a purchase the agent brings back the offer, the price and the reference, and the owner pays.

### Planned tests

1. A mission to the owner's own phone, the owner playing a service that offers two slots and then asks for a card number. Checks: AI disclosure, respect of the order, refusal of the card, final read-back, calendar event, report, latency.
2. A real service with nothing at stake, chosen by the owner, during opening hours: asking for Saturday opening times.
3. Someone the owner knows, warned beforehand, on a real errand.

Each step goes through a mission order, a go, a report and the owner's verdict before the next one.

### Limits and legal points

- Failures reported by third-party tests and taken into account in the design: an AI disclosure played to a voice menu instead of the human who picks up next, a summary that says "booked" without confirmation, an invented phone number, a reverse phishing attempt asking for a card number and a text-message code, a business that refuses to deal with an AI.
- French administrations (long waits, identity checks, refusal to talk to a third party): no published test found. Since the agent never claims to be the owner, it can obtain general information and appointments, and has no access to a personal file.
- France and the EU, according to sources read on 4 November 2026, to be confirmed by a lawyer: Article 50(1) of the AI Act requires that people know they are talking to an AI, applicable since 2 August 2026; the CNIL asks that each person be told about a recording at the start of the call; cold calling consumers without prior consent is banned since 11 August 2026; Arcep restricts which numbers automated calling systems may display, and whether a one-off agent call counts as such a system was not settled in what was read. These calls are service calls and the agent is never to be used for prospecting.

## Made with Claude Code

The pipeline was assembled, measured and documented in Claude Code sessions run by the author, and the calling agent was designed in the same way. The go gate follows the rule he applies to every message Claude Code prepares for him: nothing leaves for a human without the full text displayed and an explicit go that names the act.

## Repository layout

```
pipeline/
  bot.py              Pipecat bot: STT, end of turn, LLM, TTS, note at the end of the call
  stt_server.py       Kyutai streaming STT over a websocket
  persona.md          persona with the {owner} placeholder
  bench.py            file-based bench with per-step timings
  test_webrtc.py      WebRTC client that plays a WAV into the real bot
  make_questions.py   the five bench questions (Kokoro)
  make_scenarios.py   two multi-turn scenarios (Kokoro)
  scripts/            start and stop scripts, Windows variant of the LLM launcher
  measurements/       bench records of 5 November 2026, with and without the guard
  requirements-*.txt  versions installed in the measured setup
caller/
  mission-order.template.md     mission order, English
  mission-order.template.fr.md  mission order, French
```

Audio files are left out of the repository.

---

State verified on 5 November 2026.

Ismaël Joffroy Chandoutis, 2026. Code under the MIT license, texts under CC BY 4.0 ([LICENSE](LICENSE)).

By [Ismaël Joffroy Chandoutis](https://ismaeljoffroychandoutis.com).
