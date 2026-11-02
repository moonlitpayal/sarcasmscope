"""TOY data generator, for software tests and the demo app ONLY.

These sentences are machine-written template phrases in several scripts. They have NOT been
checked by native speakers and are NOT a research dataset. Never report results on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEX = {
    "hi": dict(pos=["वाह", "क्या बात है", "बहुत बढ़िया"],
               sit=["फिर से बारिश", "ट्रैफिक में फँस गए", "बिजली चली गई"],
               neu=["आज मौसम अच्छा है", "मैं घर जा रहा हूँ", "खाना तैयार है"]),
    "mr": dict(pos=["व्वा", "छान", "खूपच छान"],
               sit=["पुन्हा पाऊस", "ट्रॅफिकमध्ये अडकलो", "लाईट गेली"],
               neu=["आज हवामान छान आहे", "मी घरी जात आहे", "जेवण तयार आहे"]),
    "te": dict(pos=["అద్భుతం", "చాలా బాగుంది"],
               sit=["మళ్లీ వర్షం", "ట్రాఫిక్ లో ఇరుక్కున్నాం", "కరెంట్ పోయింది"],
               neu=["ఈ రోజు వాతావరణం బాగుంది", "నేను ఇంటికి వెళ్తున్నాను", "భోజనం సిద్ధంగా ఉంది"]),
    "ta": dict(pos=["அருமை", "சூப்பர்"],
               sit=["மறுபடியும் மழை", "டிராஃபிக்கில் மாட்டிக்கொண்டோம்", "மின்சாரம் போய்விட்டது"],
               neu=["இன்று வானிலை நன்றாக இருக்கிறது", "நான் வீட்டிற்கு போகிறேன்", "சாப்பாடு தயாராக உள்ளது"]),
    "bn": dict(pos=["বাহ", "দারুণ"],
               sit=["আবার বৃষ্টি", "যানজটে আটকে গেছি", "বিদ্যুৎ চলে গেল"],
               neu=["আজ আবহাওয়া ভালো", "আমি বাড়ি যাচ্ছি", "খাবার তৈরি"]),
    "hi-en": dict(pos=["wah", "kya baat hai", "mast"],
                  sit=["phir se baarish", "traffic mein fas gaye", "light chali gayi"],
                  neu=["aaj mausam achha hai", "main ghar ja raha hoon", "khana ready hai"]),
    "en": dict(pos=["great", "wonderful", "perfect"],
               sit=["another rainy Monday", "stuck in traffic again", "power cut again"],
               neu=["the weather is nice today", "I am going home", "dinner is ready"]),
}
IRONY = ["🙄", "😏", "🙃", "😒"]
HAPPY = ["😊", "😂", "👍", "😍"]


def make_toy(n_per_lang: int = 60, seed: int = 0, leak: float = 0.5, n_dups: int = 3,
             suffix: bool = False) -> pd.DataFrame:
    """leak = probability that an irony emoji is added to sarcastic rows (and happy emoji to others).

    leak=0 gives data with no emoji shortcut; leak=1 gives a strong shortcut (used to test the audit).
    suffix=True appends a random number so that rows are unique (template sentences repeat otherwise).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for lang, lex in LEX.items():
        for i in range(n_per_lang):
            label = int(i % 2 == 0)
            pos, sit, neu = (rng.choice(lex[k]) for k in ("pos", "sit", "neu"))
            punct = rng.choice(["!", ".", "..."])
            if label == 1:
                text = f"{pos}, {sit}{punct}"
                emo = rng.choice(IRONY) if rng.rand() < leak else ""
            else:
                kind = rng.randint(0, 4)
                text = [neu, f"{pos}, {neu}{punct}", sit, f"{neu}{punct}"][kind]
                emo = rng.choice(HAPPY) if rng.rand() < leak * 0.5 else ""
            if emo:
                text += " " + emo
            if suffix:
                text += f" {rng.randint(1, 9999)}"
            if lang in ("hi-en", "en") and rng.rand() < 0.15:
                text = text.replace("wah", "wahhhh").replace("great", "greaaat")
            if rng.rand() < 0.10:
                text += " #life"
            if rng.rand() < 0.05:
                text = "@friend " + text
            rows.append({"text": text, "label": label, "lang": lang})
    df = pd.DataFrame(rows)
    if n_dups:
        df = pd.concat([df, df.sample(n_dups, random_state=seed)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"toy-{i}" for i in range(len(df))])
    return df
