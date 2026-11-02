"""Look inside any dataset file before using it:  python scripts/peek_dataset.py data/raw/<file>"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = Path(sys.argv[1])
suf = p.suffix.lower()
if suf in (".xlsx", ".xls"):
    xl = pd.ExcelFile(p)
    print("Sheets:", xl.sheet_names)
    df = xl.parse(xl.sheet_names[0])
elif suf == ".csv":
    df = pd.read_csv(p, encoding="utf-8-sig")
elif suf in (".tsv", ".txt"):
    df = pd.read_csv(p, sep="\t", encoding="utf-8-sig")
else:
    df = pd.read_json(p, lines=(suf == ".jsonl"))
print("Rows, columns:", df.shape)
for c in list(df.columns)[:30]:
    ex = str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else ""
    print(f"  {c!r}: {df[c].nunique()} distinct | example: {ex}")
for c in list(df.columns)[:30]:
    if df[c].nunique() <= 15:
        print("\nValue counts for", c, "->", df[c].value_counts().to_dict())
print("\nLook for: a SARCASM label column? a language column? duplicates? Read the dataset's README and licence too.")
