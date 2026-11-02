"""Generate a realistic Marathi-English code-mixed demo corpus for testing.

This is NOT a research corpus. It is a synthetic file used only to:
  - verify that annotate-sample runs end-to-end before the real corpus arrives
  - let you test the annotation workbook pipeline on your machine

Usage:
    python scripts/make_demo_corpus.py --out data/raw/mecorpus_demo.txt --n 5000

The real corpus (L3Cube-MeCorpus, 139 MB, CC BY-NC-SA 4.0) is downloaded from:
  https://drive.google.com/file/d/1fvDEVlb1SCaxqUl3Xu5LYzAJWLzHeLy0/view?usp=sharing
Move the unzipped file to  data/raw/mecorpus.txt  and use that instead.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

# Marathi / Hinglish / English fragments typical of MeCorpus social-media posts
MARATHI_WORDS = [
    "आज", "हवामान", "छान", "आहे", "खूप", "चांगलं", "काय", "करतो", "आम्ही", "बघू",
    "होय", "नाही", "पण", "मला", "वाटतं", "खरं", "बोलतो", "ट्रॅफिक", "बघ", "लाईट",
    "पाऊस", "उशीर", "ऑफिस", "पोहोचलो", "सुंदर", "भारी", "मस्त", "व्वा", "वाह", "छे",
    "हे", "काम", "केलं", "बरं", "झालं", "एकदम", "परफेक्ट", "सरकार", "रस्ते", "पाणी",
    "बस", "रिक्षा", "मुंबई", "पुणे", "नाशिक", "वेळ", "खाणं", "जेवण", "आणि", "किती",
]
ENGLISH_WORDS = [
    "traffic", "rain", "office", "power", "cut", "great", "super", "amazing", "wow",
    "really", "very", "good", "bad", "so", "nice", "terrible", "boring", "life",
    "today", "again", "late", "stuck", "perfect", "awesome", "work", "day", "thanks",
    "government", "roads", "service", "internet", "connection", "train", "bus",
]
IRONY_PHRASES = [
    "व्वा, परत traffic jam! किती छान.",
    "wah wah, phir se power cut, mazaa aa gaya yaar!",
    "आहाहा, पुन्हा पाऊस, खूप छान हवामान!",
    "great, bus cancelled again, very helpful service!",
    "super, office late again because of amazing Pune traffic!",
    "वाह, सरकारने परत रस्ते खराब केले, खूप भारी काम!",
    "wow, internet gone again at 9 PM, perfect timing as always!",
    "wah re wah, phir se load shedding, bahut badia!",
    "अरे व्वा, परत उशीर, कामच राहिलं नाही असं वाटतं!",
    "amazing, three hours in traffic for a 10 km ride, so efficient!",
    "wah, paus ani traffic ekach veli, double fun!",
    "Great job guys, pothole on every road, very smart planning!",
    "khup chan, office fire drill on a Monday morning, exactly what we needed!",
    "व्वा, आज परत 3 तास ट्रॅफिकमध्ये, जगण्यात मजा आहे!",
    "super idea to do road repair during peak hours, brilliant!",
]
PLAIN_PHRASES = [
    "आज हवामान खूप छान आहे, मला आवडलं.",
    "office cha rasta changla ahe today.",
    "Pune traffic is a daily challenge but we manage.",
    "मला पाऊस खूप आवडतो, मन प्रसन्न होतं.",
    "khana bahut tasty tha aaj, khub enjoy kiya.",
    "bus on time today, good service!",
    "Nashik is a peaceful city compared to Mumbai.",
    "आम्ही सगळे एकत्र जेवलो, खूप मजा आली.",
    "Today completed the project deadline, feeling satisfied.",
    "woke up early, workout done, feeling fresh!",
    "रिक्षावाल्याने मीटरने नेलं, खूप बरं वाटलं.",
    "internet speed is great today, downloads fast.",
    "पुण्यात पाऊस म्हणजे मातीचा वास, खूप छान.",
    "ऑफिसमधून लवकर निघालो, traffic कमी होतं.",
    "aaj road was clear, reached in 20 minutes only!",
    "नाशिकचे द्राक्षे खूप गोड असतात.",
    "Meeting was productive today, good decisions taken.",
    "Mumbai local trains are always on time, reliable service.",
    "मला Marathi food खूप आवडतो, specially puran poli.",
    "Sunday morning chai and newspaper, perfect relaxation.",
]


def make_corpus(n: int = 5000, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    lines = []
    # Add irony phrases multiple times with slight variation
    for _ in range(max(1, n // 20)):
        for phrase in IRONY_PHRASES:
            lines.append(phrase)
    # Add plain phrases
    for _ in range(max(1, n // 15)):
        for phrase in PLAIN_PHRASES:
            lines.append(phrase)
    # Generate synthetic mixed sentences to pad up to n
    while len(lines) < n:
        use_marathi = rng.random() < 0.6
        words = []
        for _ in range(rng.randint(6, 14)):
            if use_marathi and rng.random() < 0.7:
                words.append(rng.choice(MARATHI_WORDS))
            else:
                words.append(rng.choice(ENGLISH_WORDS))
        # Add irony cues to some
        if rng.random() < 0.15:
            words.insert(0, rng.choice(["वाह", "व्वा", "wah", "wow", "great"]))
            words.append(rng.choice(["traffic", "ट्रॅफिक", "पाऊस", "power cut", "boring"]))
        lines.append(" ".join(words) + rng.choice(["", "!", ".", "...", " 🙄", " 😂"]))
    rng.shuffle(lines)
    return lines[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/raw/mecorpus_demo.txt", help="Output .txt file path")
    ap.add_argument("--n", type=int, default=5000, help="Number of lines")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = make_corpus(a.n, a.seed)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} demo lines to {p}")
    print("First 5 lines:")
    for line in lines[:5]:
        print(f"  {line}")
    print(f"\nNow run:\n  sarcasmscope annotate-sample --source {a.out} --annotators Payal Divanshi Aniket")


if __name__ == "__main__":
    main()
