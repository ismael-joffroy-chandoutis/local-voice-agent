"""Makes the five bench questions once, as WAV files in questions/, with Kokoro on CPU.
The names in the questions (Claire Martin, John) are fictional test callers."""
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline

Q = [("fr1", "f", "ff_siwis", "Bonjour, tu peux me rappeler pourquoi on garde une marge de quinze pour cent sur les disques ?"),
     ("fr2", "f", "ff_siwis", "Donne-moi une idée simple pour le dîner de ce soir, avec ce qu'on trouve dans un placard."),
     ("en1", "a", "af_heart", "Hi, can you explain in two sentences what a voice activity detector does?"),
     ("fr3", "f", "ff_siwis", "Bonjour, je suis Claire Martin, je voudrais parler à Ismaël pour un projet de film."),
     ("en2", "a", "af_heart", "Hello, this is John from the festival. Can I speak to Ismaël?")]
os.makedirs("questions", exist_ok=True)
p = {"f": KPipeline(lang_code="f", device="cpu"), "a": KPipeline(lang_code="a", device="cpu")}
for name, l, v, t in Q:
    a = np.concatenate([np.asarray(x.audio) for x in p[l](t, voice=v)])
    sf.write(f"questions/{name}.wav", (a * 32767).astype(np.int16), 24000)
    print(name, len(a) / 24000)
