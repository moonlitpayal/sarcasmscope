"""One Excel workbook per annotator, with a drop-down label column and no information about other annotators."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

LABELS = ["sarcastic", "not sarcastic", "cannot tell"]
GUIDELINES = Path(__file__).resolve().parents[3] / "docs" / "ANNOTATION_GUIDELINES.md"


def _guideline_lines() -> list[str]:
    if GUIDELINES.exists():
        return GUIDELINES.read_text(encoding="utf-8").splitlines()
    return ["Guidelines file not found. See docs/ANNOTATION_GUIDELINES.md in the project."]


def write_annotator_workbook(items: pd.DataFrame, annotator: str, path: str | Path) -> Path:
    """items needs item_id and text (nothing else is written: annotators must not see subsets or cues)."""
    wb = Workbook()
    g = wb.active
    g.title = "Guidelines"
    for i, line in enumerate(_guideline_lines(), 1):
        g.cell(row=i, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
    g.column_dimensions["A"].width = 120

    ws = wb.create_sheet("Label")
    for j, h in enumerate(["item_id", "text", "label", "note (optional)"], 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", start_color="1F3864")
    for i, r in enumerate(items[["item_id", "text"]].itertuples(index=False), 2):
        ws.cell(row=i, column=1, value=r.item_id)
        ws.cell(row=i, column=2, value=r.text).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=i, column=3).fill = PatternFill("solid", start_color="FFFF99")
    last = len(items) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Pick from the list", error="Choose: sarcastic, not sarcastic or cannot tell")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{last}")
    for col, w in zip("ABCD", (9, 90, 18, 30)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"

    pr = wb.create_sheet("Progress")
    rows = [("Annotator", annotator), ("Items", len(items)), ("Labelled so far", f"=COUNTA(Label!C2:C{last})"),
            ("Still to do", "=B2-B3"), ("Marked sarcastic", f'=COUNTIF(Label!C2:C{last},"sarcastic")'),
            ("Marked cannot tell", f'=COUNTIF(Label!C2:C{last},"cannot tell")')]
    for i, (a, b) in enumerate(rows, 1):
        pr.cell(row=i, column=1, value=a)
        pr.cell(row=i, column=2, value=b)
    pr.column_dimensions["A"].width = 22
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def read_annotator_workbook(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Label", dtype=str)
    df = df.rename(columns=lambda c: str(c).split(" ")[0].strip().lower())
    df = df[["item_id", "label"]].copy()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    return df
