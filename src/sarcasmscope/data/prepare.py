"""Dataset preparation helpers (convert a public dataset's native files into a clean CSV).

SECURITY: some datasets ship Python pickles. Unpickling arbitrary files can execute code, so
`safe_read_pickle` only accepts the exact pandas/numpy classes a plain DataFrame needs and refuses everything else.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import pandas as pd

ALLOWED_GLOBALS = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.series", "Series"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
    # names written by newer pandas versions (public aliases and the block helper)
    ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"), ("pandas", "RangeIndex"),
    ("pandas", "MultiIndex"), ("pandas._libs.internals", "_unpickle_block"),
}


def safe_read_pickle(path: str | Path) -> pd.DataFrame:
    from pandas.compat import pickle_compat

    class _Safe(pickle_compat.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in ALLOWED_GLOBALS:
                raise pickle.UnpicklingError(
                    f"Refusing to load {path}: it references {module}.{name}, which is not a plain pandas/numpy "
                    f"DataFrame component. Only open pickles you trust, in an isolated environment.")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        obj = _Safe(io.BytesIO(f.read())).load()
    if not isinstance(obj, pd.DataFrame):
        raise pickle.UnpicklingError(f"{path} did not contain a DataFrame")
    return obj


def _read_lines(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return txt.split("\n")


def prepare_masac(repo_dir: str | Path, out_csv: str | Path, context_turns: int = 2) -> dict:
    """Build one clean CSV from the MaSaC / MSH-COMICS repository (Hindi-English, sarcasm + humour).

    Text comes from Raw_Utterances/Utterances.txt (clean), labels from Pickles/Dataset.p, and the two are aligned
    row by row; alignment is verified against the dialogue labels and speaker names before anything is written.
    Output columns: id, speaker, text, sarcasm, humour, lang, dialogue, context (previous turns of the same dialogue).
    """
    repo = Path(repo_dir)
    pk = safe_read_pickle(repo / "Data" / "Pickles" / "Dataset.p")
    raw = _read_lines(repo / "Data" / "Raw_Utterances" / "Utterances.txt")
    raw = [l for l in raw if l.strip() != ""]
    ep = [e for e in _read_lines(repo / "Data" / "Raw_Utterances" / "Episodewise_dialoguelabels.txt") if e.strip()]
    ep = ep[1:] if ep and ep[0].strip().lower().startswith("episode") else ep
    if not (len(raw) == len(ep) == len(pk)):
        raise ValueError(f"Row counts differ: utterances {len(raw)}, dialogue labels {len(ep)}, labelled table {len(pk)}")
    speakers, texts = [], []
    for line in raw:
        sp, _, tx = line.partition(":")
        speakers.append(sp.strip())
        texts.append(tx.strip())
    spk_match = (pd.Series(speakers).str.lower().values == pk["Speaker"].astype(str).str.strip().str.lower().values).mean()
    ep_match = (pd.Series(ep).values == pk["Episode_label"].values).mean()
    if spk_match < 0.99 or ep_match < 0.99:
        raise ValueError(f"Raw text and labelled table are not aligned (speaker match {spk_match:.3f}, "
                         f"dialogue match {ep_match:.3f}). Not writing anything.")
    df = pd.DataFrame({"speaker": speakers, "text": texts, "sarcasm": pk["Sarcasm"].astype(int).values,
                       "humour": pk["Humour"].astype(int).values, "dialogue": ep})
    run_id = (df["dialogue"] != df["dialogue"].shift()).cumsum()
    df["dialogue"] = df["dialogue"] + "#" + run_id.astype(str)         # contiguous run = one dialogue instance
    ctx = []
    for _, part in df.groupby("dialogue", sort=False):
        lines = [f"{s}: {t}" for s, t in zip(part["speaker"], part["text"])]
        for i in range(len(lines)):
            ctx.append(" || ".join(lines[max(0, i - context_turns):i]))
    df["context"] = ctx
    df["lang"] = "hi-en"
    df.insert(0, "id", [f"masac-{i}" for i in range(len(df))])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return {"rows": len(df), "dialogues": int(df["dialogue"].nunique()), "sarcastic": int(df["sarcasm"].sum()),
            "speaker_match": float(spk_match), "dialogue_match": float(ep_match), "out": str(out_csv)}
