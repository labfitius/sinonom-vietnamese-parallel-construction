"""Builder for optimize_eval.ipynb: reproducible Task 2 evaluation & visual proof notebook.
Reads optimize/improved/ned_summary.csv, optimize/improved/<book>.xlsx, and optimize/Yale/ images.

Usage: python scripts/build_optimize_eval_notebook.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

CELLS = [
    ("md", """# Task 2 Evaluation: Improving Handwritten Sino-Nom OCR Accuracy Beyond Teacher Baseline (`optimize/`)

## Architecture Overview
To beat the teacher's strong baseline (**97.13% – 98.38%** across 4 Yale handwritten Sino-Nom books), our audit proved that re-ranking within the teacher's 3-model top-10 candidates only recovers ~3% of remaining errors (~+0.1%). Therefore, we designed a **zero-leakage 4-pillar optimization pipeline**:

1. **Dictionary v2 (`core/nomdict.py`, `data/dicts/nomdict_v2.pkl`)**: Fixed Telex-undecoded readings in `nom_viet.dict.yaml` + merged `Unihan kVietnamese` + tone-placement-agnostic matching (`syl_key`), raising exact coverage on teacher-accepted pairs from **72.96% to 87.35%**.
2. **5-Fold Page-Grouped ConvNeXt-Tiny Classifier (`scripts/opt_train_classifier.py`)**: Trained `convnext_tiny.fb_in22k_ft_in1k` (96x96, 3,574 classes) strictly grouped by page (zero page leakage!) on ~27,637 high-confidence crops, achieving **~87.8% pure visual top-1** and **~93.7% top-5** without any phonetic hint.
3. **Full-Vocabulary Sound-Constrained Decoding (`scripts/opt_ensemble_improve.py`)**: Breaks the teacher's top-10 ceiling by scoring **every** dictionary character matching the aligned Vietnamese syllable using out-of-fold ConvNeXt probabilities.
4. **Independent 4th OCR Voter (KimHanNom FIT HCMUS API) + QN Typo Guard**: Queries KimHanNom Cloud OCR on all 690 error/mismatch lines to repair missing/extra characters and corroborate fixes, while flagging lines where all 4 visual models strongly agree against a noisy Vietnamese syllable."""),

    ("code", """import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Image as IPImage

ROOT = Path.cwd().resolve()
OPT_IMP = ROOT / "optimize" / "improved"
OPT_YALE = ROOT / "optimize" / "Yale"

summary = pd.read_csv(OPT_IMP / "ned_summary.csv", encoding="utf-8-sig")
display(summary)"""),

    ("md", "## 1. Before vs After Comparison & Error Reduction Breakdown"),

    ("code", """import matplotlib.pyplot as plt

plot_df = summary[summary["Book"] != "OVERALL (4 BOOKS)"].copy()
short_names = [b[:22] + "..." if len(b) > 25 else b for b in plot_df["Book"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(short_names))
w = 0.36
axes[0].bar(x - w/2, plot_df["Teacher_Acc_%"], w, label="Teacher Baseline", color="#5c7cfa")
axes[0].bar(x + w/2, plot_df["Improved_Acc_%"], w, label="Our 4-Pillar Improved", color="#2f9e44")
axes[0].set_ylim(95.0, 100.0)
axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, rotation=15, ha="right")
axes[0].set_ylabel("Character Accuracy (%)")
axes[0].set_title("Teacher Baseline vs Improved Accuracy")
axes[0].legend()
for i, (t, n) in enumerate(zip(plot_df["Teacher_Acc_%"], plot_df["Improved_Acc_%"])):
    axes[0].text(i - w/2, t + 0.08, f"{t:.2f}%", ha="center", fontsize=8)
    axes[0].text(i + w/2, n + 0.08, f"{n:.2f}%", ha="center", fontsize=8, fontweight="bold")

axes[1].bar(short_names, plot_df["Error_Reduction_%"], color="#f08c00")
axes[1].set_xticklabels(short_names, rotation=15, ha="right")
axes[1].set_ylabel("Error Reduction (%)")
axes[1].set_title("Percentage of Teacher Remaining Errors Repaired")
for i, v in enumerate(plot_df["Error_Reduction_%"]):
    axes[1].text(i, v + 0.8, f"-{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()"""),

    ("md", "## 2. Sample Repaired Lines & Vietnamese Typo Flags"),

    ("code", """books = [
    "Ca Trù Thể Cách, Nôm",
    "Nhi thap tu nu tac dien am digcoll_12920",
    "Thánh Du Quảng Huân Diên Nghĩa, Nôm",
    "Tây Du Truyên, Nôm",
]
samples = []
typos = []
for b in books:
    df = pd.read_excel(OPT_IMP / f"{b}.xlsx", dtype=str).fillna("")
    chg = df[df["Changed"] == "YES"].head(5).copy()
    chg.insert(0, "Book", b[:20])
    samples.append(chg[["Book", "Patch ID", "Vietnamese", "Label_Teacher", "Label", "Change_Source"]])
    tf = df[df["QN_Typo_Flag"] != ""].head(4).copy()
    tf.insert(0, "Book", b[:20])
    typos.append(tf[["Book", "Patch ID", "Vietnamese", "Label", "QN_Typo_Flag"]])

print("=== Sample Repaired Lines (Teacher -> Improved) ===")
display(pd.concat(samples, ignore_index=True))

print("\\n=== Sample Suspected Vietnamese QN Typos (Protected by 4-Model Visual Consensus) ===")
display(pd.concat(typos, ignore_index=True))"""),

    ("md", "## 3. Visual Proof on Repaired Line Crops"),

    ("code", """# Render visual side-by-side crops for first 4 repaired lines of Ca Tru The Cach
book = "Ca Trù Thể Cách, Nôm"
df = pd.read_excel(OPT_IMP / f"{book}.xlsx", dtype=str).fillna("")
chg = df[df["Changed"] == "YES"].head(4)
img_dir = OPT_YALE / book / "images"
lab = {}
for l in (img_dir / "Label.txt").read_text(encoding="utf-8").splitlines():
    k, v = l.split("\\t", 1)
    lab[Path(k).stem] = json.loads(v)

for _, r in chg.iterrows():
    pid = r["Patch ID"]
    page, li = pid.rsplit(".", 1)
    li = int(li)
    if page not in lab or li >= len(lab[page]):
        continue
    im = cv2.imdecode(np.fromfile(str(img_dir / f"{page}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    pts = np.array(lab[page][li]["points"])
    x0, y0 = np.floor(pts.min(0)).astype(int); x1, y1 = np.ceil(pts.max(0)).astype(int)
    crop = im[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{pid} | QN: {r['Vietnamese']}")
    print(f"  Teacher:  {r['Label_Teacher']}  ({r['Correctness_Teacher']})")
    print(f"  Improved: {r['Label']}  ({r['Correctness']}) | {r['Change_Source']}")
    if ok:
        display(IPImage(data=buf.tobytes(), width=700))"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in CELLS]
    out = ROOT / "optimize_eval.ipynb"
    nbf.write(nb, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
