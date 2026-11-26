"""Comprehensive content-aware audit across all 24 books in output/ (after reprocessing).
Writes reports/audit_output_24books_after.csv and prints summary table.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import OUTPUT_DIR
from core.evaluator import AlignmentEvaluator

_HAN = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF\uE000-\uF8FF\U00020000-\U0003FFFF\U000F0000-\U0010FFFF]")
_PH = re.compile(r"\.\d{3,}\.\d{1,3}$")


def load_tsv(p: Path):
    d = {}
    if not p.exists():
        return d
    for l in p.read_text(encoding="utf-8").splitlines():
        if "\t" in l:
            k, v = l.split("\t", 1)
            try:
                d[Path(k).name] = json.loads(v)
            except Exception:
                d[Path(k).name] = []
    return d


def main():
    ev_eng = AlignmentEvaluator()
    rows = []
    for o in sorted(d for d in OUTPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")):
        r = {"book": o.name}
        for f in ("ch", "vi", "misc", "blank"):
            r[f] = len(list((o / f).glob("*.jpg"))) if (o / f).is_dir() else 0
        ch = load_tsv(o / "ch" / "Label.txt")
        vi = load_tsv(o / "vi" / "Label.txt")
        chb = [x for v in ch.values() for x in v]
        vib = [x for v in vi.values() for x in v]
        r["ch_boxes"] = len(chb)
        r["vi_boxes"] = len(vib)
        r["ch_real%"] = round(100 * np.mean([bool(_HAN.search(str(x.get("transcription", "")))) for x in chb]), 1) if chb else 0.0
        r["vi_real%"] = round(100 * np.mean([bool(str(x.get("transcription", "")).strip()) and not _PH.search(str(x.get("transcription", ""))) and not str(x.get("transcription", "")).startswith(o.name) for x in vib]), 1) if vib else 0.0
        csv = o / f"{o.name}.csv"
        if csv.exists():
            df = pd.read_csv(csv, encoding="utf-8-sig", dtype=str).fillna("")
            r["csv_rows"] = len(df)
            r["nom_fill%"] = round(100 * df["SinoNom_Char"].str.strip().ne("").mean(), 1) if len(df) else 0.0
            r["qn_fill%"] = round(100 * df["ChuQN_txt"].str.strip().ne("").mean(), 1) if len(df) else 0.0
            ev = ev_eng.evaluate_dataset(df.to_dict("records"))
            r["sim%"] = round(ev["similarity_percentage"], 2) if ev["total_chars"] else 0.0
            r["passed"] = bool(ev["is_passed"] and r["nom_fill%"] >= 95 and r["qn_fill%"] >= 95)
        imgs = sorted((o / "ch").glob("*.jpg")) or sorted((o / "vi").glob("*.jpg"))
        r["img_size"] = "x".join(map(str, Image.open(imgs[0]).size)) if imgs else "-"
        rows.append(r)

    df_out = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(df_out.to_string(index=False))
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    df_out.to_csv(rep / "audit_output_24books_after.csv", index=False, encoding="utf-8-sig")
    n_pass = int(df_out["passed"].sum())
    print(f"\nSUMMARY: {n_pass}/{len(df_out)} books currently PASS strict content + >=95% similarity.")


if __name__ == "__main__":
    main()
