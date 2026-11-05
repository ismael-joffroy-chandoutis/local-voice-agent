"""Makes two multi-turn scenarios (a caller leaving a message, the owner dictating notes),
as WAV files in questions/, to be played into the real bot with test_webrtc.py.
Names and the phone number are fictional; the number is spoken as words."""
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline

p = KPipeline(lang_code="f", device="cpu")
SR = 24000


def say(t):
    return np.concatenate([np.asarray(x.audio) for x in p(t, voice="ff_siwis")])


def scenario(name, lines):
    parts = []
    for t, pause in lines:
        parts += [say(t), np.zeros(int(pause * SR), dtype=np.float32)]
    a = np.concatenate(parts)
    sf.write(f"questions/{name}.wav", (a * 32767).astype(np.int16), SR)
    print(name, round(len(a) / SR, 1))


os.makedirs("questions", exist_ok=True)
scenario("call-message", [
    ("Bonjour, je suis Claire Martin, de la société Lumen Films.", 10),
    ("Je vous appelle pour proposer un projet de court métrage à Ismaël, pour le printemps prochain.", 10),
    ("Mon numéro, c'est le zéro six, douze, trente-quatre, cinquante-six, soixante-dix-huit.", 12),
    ("Oui, c'est bien ça. Merci, au revoir.", 6)])
scenario("dictation", [
    ("Bonjour, c'est Ismaël.", 7),
    ("Note : rappeler la monteuse demain matin à propos du montage.", 7),
    ("Et ajoute : commander du papier pour l'imprimante avant vendredi.", 7),
    ("Relis-moi mes notes, s'il te plaît.", 12)])
