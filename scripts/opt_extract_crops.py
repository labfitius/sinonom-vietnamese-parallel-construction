"""Extract per-character crops + metadata from the teacher's optimize/ data.

For each Alignment row (book.page.line) we take the char boxes of char_det.txt whose centre lies
inside the line polygon of Label.txt, sort them along the reading direction (left->right for
these horizontal Yale lines), and keep the row when #boxes == #label chars == #syllables.

Output: optimize/_work/crops_<book>.npz with
  img (N,64,64) uint8, label, syl, corr (0/1), page, line, pos, cands (json: 3 models top-10)
Usage: python scripts/opt_extract_crops.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
OPT = ROOT / "optimize"
WORK = OPT / "_work"
SIZE = 64
MODELS = ["char_det.txt", "char_det.txt.ConvNeXt", "char_det.txt.ConvNeXtFinetune"]


def load_tsv(p):
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        k, v = l.split("\t", 1)
        out[Path(k).stem] = json.loads(v)
    return out


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


def square_crop(g, pts, pad=4):
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    x0, y0 = max(x0, 0), max(y0, 0)
    c = g[y0:y1, x0:x1]
    h, w = c.shape
    s = max(h, w)
    canvas = np.full((s, s), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def assign_lines(lines, chars):
    """Return list per line of char indices, sorted left->right."""
    res = [[] for _ in lines]
    polys = [np.array(l["points"]) for l in lines]
    for i, c in enumerate(chars):
        cx, cy = np.array(c["points"]).mean(0)
        best, bd = None, 1e9
        for j, p in enumerate(polys):
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            if x0 - 5 <= cx <= x1 + 5 and y0 - 5 <= cy <= y1 + 5:
                d = abs(cy - (y0 + y1) / 2)
                if d < bd:
                    best, bd = j, d
        if best is not None:
            res[best].append((cx, i))
    return [[i for _, i in sorted(r)] for r in res]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    for bdir in sorted((OPT / "Yale").iterdir()):
        book = bdir.name
        img_dir = bdir / "images"
        lab = load_tsv(img_dir / "Label.txt")
        mods = [load_tsv(img_dir / m) for m in MODELS]
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str)
        df[["page", "line"]] = df["Patch ID"].str.extract(r"^(.*)\.(\d{3})$")
        rec = {k: [] for k in ["img", "label", "syl", "corr", "page", "line", "pos", "cands"]}
        kept = skipped = 0
        for page, g in df.groupby("page", sort=False):
            if page not in lab:
                continue
            gray = imread(img_dir / f"{page}.jpg")
            per_line = assign_lines(lab[page], mods[0].get(page, []))
            for _, r in g.iterrows():
                li = int(r["line"])
                label = str(r["Label"]) if r["Label"] == r["Label"] else ""
                syl = str(r["Vietnamese"]).split()
                corr = str(r["Correctness"]).replace(".0", "").ljust(len(label), "0")
                idx = per_line[li] if li < len(per_line) else []
                if not label or len(idx) != len(label) or len(syl) != len(label):
                    skipped += 1
                    continue
                kept += 1
                for pos, ci in enumerate(idx):
                    c0 = mods[0][page][ci]
                    rec["img"].append(square_crop(gray, np.array(c0["points"])))
                    rec["label"].append(label[pos]); rec["syl"].append(syl[pos].lower())
                    rec["corr"].append(int(corr[pos] == "1"))
                    rec["page"].append(page); rec["line"].append(li); rec["pos"].append(pos)
                    cands = [[m[page][ci]["labels"], m[page][ci]["scores"]] if ci < len(m.get(page, [])) else [[], []] for m in mods]
                    rec["cands"].append(json.dumps(cands, ensure_ascii=False))
        np.savez_compressed(WORK / f"crops_{book}.npz", img=np.stack(rec["img"]),
                            **{k: np.array(v) for k, v in rec.items() if k != "img"})
        print(f"{book}: rows kept={kept} skipped={skipped} chars={len(rec['label'])}")


if __name__ == "__main__":
    main()
