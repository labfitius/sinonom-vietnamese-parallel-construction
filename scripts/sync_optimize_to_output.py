"""Tier A & Tier B Reprocessor (Strict compliance with guide/Quy_dinh_do_an.pdf):

Tier A (4 books from optimize/improved/):
  - Ca Trù Thể Cách, Nôm_Y
  - Nhi thap tu nu tac dien am digcoll_12920_Y
  - Thánh Du Quảng Huân Diên Nghĩa, Nôm
  - Tây Du Truyên, Nôm_Y
  Renders PDFs at true 300 DPI, routes blank/misc/ch/vi properly, maps our improved Task 2
  Hán Nôm + Quốc ngữ labels onto 300 DPI 4-point PPOCRLabel boxes, and writes 4-column CSV +
  character-level Black/Red CellRichText XLSX.

Tier B (5 baseline-aligned books):
  - Ca Trù Thê Cách, AB 160, Nôm_Y
  - Ca Trù Thế Cách, AB 431, Nôm_Y
  - Ca Trù Thế Cách, AB 456, Nôm_Y
  - ThuDạLữHoàiNgâm_秋夜旅懷吟
  - Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch
  Re-renders at true 300 DPI, preserves aligned texts, cleans per guide/05, and exports
  character-level Black/Red CellRichText XLSX + compliant PPOCRLabel ch/ and vi/.

Usage: python scripts/sync_optimize_to_output.py --tier all
"""
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.aligner import SentenceAligner  # noqa: E402
from core.cleaner import PPOCRLabelFormatter, TextCleaner  # noqa: E402
from core.config import DATASETS_MIDTERM_DIR, OUTPUT_DIR  # noqa: E402
from core.evaluator import AlignmentEvaluator  # noqa: E402
from core.exporter import CorpusExporter  # noqa: E402
from core.preprocess import DocumentPreprocessor, cv2_imread_unicode  # noqa: E402

OPT = ROOT / "optimize"
OPT_IMP = OPT / "improved"
PDF_DIR = DATASETS_MIDTERM_DIR / "pdf"
CSV_DIR = DATASETS_MIDTERM_DIR / "csv"

TIER_A_MAP = {
    "Ca Trù Thể Cách, Nôm_Y": "Ca Trù Thể Cách, Nôm",
    "Nhi thap tu nu tac dien am digcoll_12920_Y": "Nhi thap tu nu tac dien am digcoll_12920",
    "Thánh Du Quảng Huân Diên Nghĩa, Nôm": "Thánh Du Quảng Huân Diên Nghĩa, Nôm",
    "Tây Du Truyên, Nôm_Y": "Tây Du Truyên, Nôm",
}

TIER_B_BOOKS = [
    ("Ca Trù Thê Cách, AB 160, Nôm_Y", "T2bw"),
    ("Ca Trù Thế Cách, AB 431, Nôm_Y", "T2bw"),
    ("Ca Trù Thế Cách, AB 456, Nôm_Y", "T2bw"),
    ("ThuDạLữHoàiNgâm_秋夜旅懷吟", "Tnom"),
    ("Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch", "Tqn"),
]


def load_tsv(p: Path):
    out = {}
    if not p.exists():
        return out
    for l in p.read_text(encoding="utf-8").splitlines():
        if "\t" in l:
            k, v = l.split("\t", 1)
            out[Path(k).stem] = json.loads(v)
    return out


def rescale_poly(pts, sx: float, sy: float, w: int, h: int):
    out = []
    for x, y in pts:
        nx = int(round(max(0, min(w - 1, x * sx))))
        ny = int(round(max(0, min(h - 1, y * sy))))
        out.append([nx, ny])
    return out


def derive_vi_box_from_ch(ch_pts, next_ch_pts, img_w: int, img_h: int, seg_vi_boxes: list):
    """Find best matching detected vi box right below ch_pts; fallback to geometric strip below ch_pts."""
    p = np.array(ch_pts)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    lh = max(30, y1 - y0)
    y_limit = int(np.array(next_ch_pts).min(0)[1]) if next_ch_pts is not None else min(img_h - 1, int(y1 + lh * 1.4))
    y_mid_target = (y1 + y_limit) / 2.0

    best = None
    best_dist = 1e9
    for vb in seg_vi_boxes:
        vp = np.array(vb["points"])
        vy0, vy1 = vp[:, 1].min(), vp[:, 1].max()
        vyc = (vy0 + vy1) / 2.0
        # Must lie below ch box top and above next ch box
        if vyc > y0 + lh * 0.45 and vyc < y_limit + 15:
            d = abs(vyc - y_mid_target)
            if d < best_dist:
                best_dist = d
                best = vb["points"]
    if best is not None:
        return [[int(round(x)), int(round(y))] for x, y in best]

    # Geometric strip directly underneath the Nôm line (standard Yale T2bw layout)
    vy0 = min(img_h - 10, int(y1 + 2))
    vy1 = min(img_h - 1, max(vy0 + 28, int(min(y_limit - 4, y1 + lh * 0.95))))
    vx0 = max(0, int(x0 - 10))
    vx1 = min(img_w - 1, int(x1 + 10))
    return [[vx0, vy0], [vx1, vy0], [vx1, vy1], [vx0, vy1]]


def process_tier_a_book(out_title: str, opt_name: str):
    print("=" * 80)
    print(f"[TIER A] Rebuilding {out_title} from improved Task 2 ({opt_name})")
    print("=" * 80)

    pdf_path = PDF_DIR / f"{out_title}.pdf"
    imp_xlsx = OPT_IMP / f"{opt_name}.xlsx"
    if not imp_xlsx.exists():
        imp_xlsx = OPT / "Alignment" / f"{opt_name}.xlsx"
        print(f"  [WARN] Using teacher Alignment (improved xlsx not yet generated): {imp_xlsx}")

    t_img_dir = OPT / "Yale" / opt_name / "images"
    t_labels = load_tsv(t_img_dir / "Label.txt")

    df_imp = pd.read_excel(imp_xlsx, dtype=str)
    # Group improved rows by 3-digit page number
    rows_by_page = defaultdict(list)
    for _, r in df_imp.iterrows():
        pid = str(r["Patch ID"])
        m = re.search(r"\.(\d{3})\.(\d{3})$", pid)
        if not m:
            continue
        pnum = int(m.group(1))
        lidx = int(m.group(2))
        rows_by_page[pnum].append({
            "line_idx": lidx,
            "patch_id": pid,
            "nom": TextCleaner.clean_nom(str(r["Label"]) if r["Label"] == r["Label"] else ""),
            "vi": TextCleaner.clean_quocngu(str(r["Vietnamese"]) if r["Vietnamese"] == r["Vietnamese"] else ""),
        })
    for pnum in rows_by_page:
        rows_by_page[pnum].sort(key=lambda x: x["line_idx"])

    book_out_dir = OUTPUT_DIR / out_title
    # Clean old ch/vi/misc/blank so no stale low-res images remain
    for sub in ("ch", "vi", "misc", "blank", "extracted_all"):
        sd = book_out_dir / sub
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)

    book_cfg = {"title": out_title, "layout_type": "T2bw"}
    prep = DocumentPreprocessor(book_cfg, OUTPUT_DIR)
    extracted = prep.split_and_extract_images(str(pdf_path))
    total_pages = len(extracted)
    print(f"  Rendered {total_pages} pages at 300 DPI.")

    ch_labels = {}
    vi_labels = {}
    aligned_records = []
    n_blank = n_misc = 0

    for page_num, img_name, img_path in extracted:
        if page_num in rows_by_page:
            shutil.copy2(str(img_path), str(prep.dir_ch / img_name))
            shutil.copy2(str(img_path), str(prep.dir_vi / img_name))

            img = cv2_imread_unicode(img_path)
            ih, iw = img.shape[:2]

            # Match teacher image size for exact coordinate rescaling
            t_stem = f"{opt_name}.{page_num:03d}"
            t_jpg = t_img_dir / f"{t_stem}.jpg"
            t_boxes = t_labels.get(t_stem, [])
            if t_jpg.exists():
                t_im = cv2_imread_unicode(t_jpg)
                th, tw = t_im.shape[:2]
                sx, sy = iw / tw, ih / th
            else:
                sx, sy = 1.0, 1.0

            # Detect vi boxes on the 300 DPI page for tight QN boxes
            seg = prep.segment_t2b_lines(img_path)
            seg_vi = seg.get("vi", [])

            p_rows = rows_by_page[page_num]
            scaled_ch = [
                rescale_poly(t_boxes[r["line_idx"]]["points"], sx, sy, iw, ih)
                if r["line_idx"] < len(t_boxes) else [[int(iw * 0.15), int(ih * 0.1)], [int(iw * 0.85), int(ih * 0.1)], [int(iw * 0.85), int(ih * 0.15)], [int(iw * 0.15), int(ih * 0.15)]]
                for r in p_rows
            ]

            ch_page_boxes = []
            vi_page_boxes = []
            stem_no_jpg = Path(img_name).stem
            for idx_in_p, (r, ch_pts) in enumerate(zip(p_rows, scaled_ch), start=1):
                nxt = scaled_ch[idx_in_p] if idx_in_p < len(scaled_ch) else None
                vi_pts = derive_vi_box_from_ch(ch_pts, nxt, iw, ih, seg_vi)
                ch_page_boxes.append({"transcription": r["nom"], "points": ch_pts})
                vi_page_boxes.append({"transcription": r["vi"], "points": vi_pts})
                aligned_records.append({
                    "img_id": img_name,
                    "sentence_id": f"{stem_no_jpg}.{idx_in_p}",
                    "SinoNom_Char": r["nom"],
                    "ChuQN_txt": r["vi"],
                })

            ch_labels[img_name] = ch_page_boxes
            vi_labels[img_name] = vi_page_boxes
        else:
            pcls = DocumentPreprocessor.classify_page(img_path, page_num, total_pages)
            if pcls == "blank":
                shutil.copy2(str(img_path), str(prep.dir_blank / img_name))
                n_blank += 1
            else:
                shutil.copy2(str(img_path), str(prep.dir_misc / img_name))
                n_misc += 1

    prep.cleanup_cache()
    shutil.copy2(str(pdf_path), str(book_out_dir / f"{out_title}.pdf"))

    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_ch, "ch", ch_labels)
    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_vi, "vi", vi_labels)

    ev = AlignmentEvaluator().evaluate_dataset(aligned_records)
    CorpusExporter.export_csv(aligned_records, book_out_dir / f"{out_title}.csv")
    CorpusExporter.export_xlsx(aligned_records, ev["row_evaluations"], book_out_dir / f"{out_title}.xlsx")
    print(f"  Pages: ch={len(ch_labels)} vi={len(vi_labels)} blank={n_blank} misc={n_misc}")
    print(f"  Official Similarity: {ev['similarity_percentage']:.2f}% ({ev['total_black_chars']}/{ev['total_chars']} chars, PASSED={ev['is_passed']})")


def fit_boxes_to_count(boxes: list, target_n: int, vertical: bool = False) -> list:
    """Merge adjacent split fragments or drop tiny noise boxes so len(boxes) == target_n."""
    if not boxes or target_n <= 0 or len(boxes) == target_n:
        return boxes
    bxs = [dict(b) for b in boxes]

    def bounds(b):
        p = np.array(b["points"])
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    # First merge overlapping/same-line fragments (same y-band for horizontal, same x-band for vertical)
    while len(bxs) > target_n:
        # Find closest pair along reading axis
        best_i, best_gap = -1, 1e18
        for i in range(len(bxs) - 1):
            x0a, y0a, x1a, y1a = bounds(bxs[i])
            x0b, y0b, x1b, y1b = bounds(bxs[i + 1])
            if vertical:
                gap = abs(((x0a + x1a) / 2.0) - ((x0b + x1b) / 2.0))
            else:
                gap = max(0.0, y0b - y1a) + 0.15 * abs(((y0a + y1a) / 2.0) - ((y0b + y1b) / 2.0))
            if gap < best_gap:
                best_gap = gap
                best_i = i
        if best_i >= 0 and best_gap < 45:
            x0a, y0a, x1a, y1a = bounds(bxs[best_i])
            x0b, y0b, x1b, y1b = bounds(bxs[best_i + 1])
            nx0, ny0 = int(min(x0a, x0b)), int(min(y0a, y0b))
            nx1, ny1 = int(max(x1a, x1b)), int(max(y1a, y1b))
            bxs[best_i] = {"points": [[nx0, ny0], [nx1, ny0], [nx1, ny1], [nx0, ny1]]}
            bxs.pop(best_i + 1)
        else:
            # Drop smallest-area noise box (page number / ink fleck)
            areas = [(bounds(b)[2] - bounds(b)[0]) * (bounds(b)[3] - bounds(b)[1]) for b in bxs]
            bxs.pop(int(np.argmin(areas)))
    return bxs


def process_tier_b_book(title: str, layout_type: str):
    print("=" * 80)
    print(f"[TIER B] Reprocessing {title} ({layout_type}) at 300 DPI + RichText XLSX")
    print("=" * 80)
    pdf_path = PDF_DIR / f"{title}.pdf"
    csv_path = CSV_DIR / f"{title}.csv"
    if not pdf_path.exists() or not csv_path.exists():
        print(f"  [SKIP] Missing pdf or csv for {title}")
        return

    book_out_dir = OUTPUT_DIR / title
    for sub in ("ch", "vi", "misc", "blank", "extracted_all"):
        sd = book_out_dir / sub
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)

    book_cfg = {"title": title, "layout_type": layout_type}
    prep = DocumentPreprocessor(book_cfg, OUTPUT_DIR)
    extracted = prep.split_and_extract_images(str(pdf_path))
    total_pages = len(extracted)

    aligner = SentenceAligner(book_cfg, csv_path)
    aligned_records = aligner.align_book_data()
    # Ensure strict cleaning per guide/05
    for r in aligned_records:
        r["SinoNom_Char"] = TextCleaner.clean_nom(str(r.get("SinoNom_Char", "") or ""))
        r["ChuQN_txt"] = TextCleaner.clean_quocngu(str(r.get("ChuQN_txt", "") or ""))

    page_to_recs = defaultdict(list)
    for r in aligned_records:
        page_to_recs[r["img_id"]].append(r)

    ch_labels, vi_labels = {}, {}
    n_blank = n_misc = 0

    for page_num, img_name, img_path in extracted:
        recs = page_to_recs.get(img_name, [])
        if layout_type == "Tqn":
            # Monolingual Vietnamese PDF companion
            shutil.copy2(str(img_path), str(prep.dir_vi / img_name))
            seg = prep.segment_t2b_lines(img_path)
            vi_boxes = fit_boxes_to_count(seg.get("vi", []) or seg.get("ch", []), len(recs), vertical=False)
            vi_labels[img_name] = [
                {"transcription": recs[i]["ChuQN_txt"], "points": b["points"]}
                for i, b in enumerate(vi_boxes[:len(recs)])
            ]
        elif layout_type == "Tnom":
            if recs:
                shutil.copy2(str(img_path), str(prep.dir_ch / img_name))
                ch_boxes = fit_boxes_to_count(prep.segment_vertical_nom_columns(img_path), len(recs), vertical=True)
                ch_labels[img_name] = [
                    {"transcription": recs[i]["SinoNom_Char"], "points": b["points"]}
                    for i, b in enumerate(ch_boxes[:len(recs)])
                ]
            else:
                pcls = DocumentPreprocessor.classify_page(img_path, page_num, total_pages)
                shutil.copy2(str(img_path), str((prep.dir_blank if pcls == "blank" else prep.dir_misc) / img_name))
                n_blank += (pcls == "blank"); n_misc += (pcls != "blank")
        else:
            if recs:
                shutil.copy2(str(img_path), str(prep.dir_ch / img_name))
                shutil.copy2(str(img_path), str(prep.dir_vi / img_name))
                seg = prep.segment_t2b_lines(img_path)
                ch_boxes = fit_boxes_to_count(seg.get("ch", []), len(recs), vertical=False)
                vi_boxes = fit_boxes_to_count(seg.get("vi", []), len(recs), vertical=False)
                ch_labels[img_name] = [
                    {"transcription": recs[i]["SinoNom_Char"], "points": b["points"]}
                    for i, b in enumerate(ch_boxes[:len(recs)])
                ]
                vi_labels[img_name] = [
                    {"transcription": recs[i]["ChuQN_txt"], "points": b["points"]}
                    for i, b in enumerate(vi_boxes[:len(recs)])
                ]
            else:
                pcls = DocumentPreprocessor.classify_page(img_path, page_num, total_pages)
                shutil.copy2(str(img_path), str((prep.dir_blank if pcls == "blank" else prep.dir_misc) / img_name))
                n_blank += (pcls == "blank"); n_misc += (pcls != "blank")

    prep.cleanup_cache()
    shutil.copy2(str(pdf_path), str(book_out_dir / f"{title}.pdf"))

    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_ch, "ch", ch_labels)
    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_vi, "vi", vi_labels)

    ev = AlignmentEvaluator().evaluate_dataset(aligned_records)
    CorpusExporter.export_csv(aligned_records, book_out_dir / f"{title}.csv")
    CorpusExporter.export_xlsx(aligned_records, ev["row_evaluations"], book_out_dir / f"{title}.xlsx")
    print(f"  Pages: ch={len(ch_labels)} vi={len(vi_labels)} blank={n_blank} misc={n_misc}")
    print(f"  Official Similarity: {ev['similarity_percentage']:.2f}% ({ev['total_black_chars']}/{ev['total_chars']} chars, PASSED={ev['is_passed']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B", "all"], default="all")
    args = ap.parse_args()

    if args.tier in ("A", "all"):
        for out_t, opt_n in TIER_A_MAP.items():
            process_tier_a_book(out_t, opt_n)
    if args.tier in ("B", "all"):
        for title, ltype in TIER_B_BOOKS:
            process_tier_b_book(title, ltype)


if __name__ == "__main__":
    main()
