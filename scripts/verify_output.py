"""
Strict Verification Script for Project Deliverables
Validates all rules specified in Quy định đồ án.
"""
import os
import re
import sys
import json
import argparse
from typing import Optional
import pandas as pd
import openpyxl
from openpyxl.cell.rich_text import CellRichText
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import BOOKS, OUTPUT_DIR, COLOR_RED, COLOR_BLACK, MIN_SIMILARITY_THRESHOLD
from core.evaluator import AlignmentEvaluator

_HAN_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF\uE000-\uF8FF\U00020000-\U0003FFFF\U000F0000-\U0010FFFF]")
_PLACEHOLDER_RE = re.compile(r"\.\d{3,}\.\d{1,3}$")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def resolve_verify_title(book_key: str) -> Optional[str]:
    """Resolve book title for verification from BOOKS or OUTPUT_DIR."""
    book = BOOKS.get(book_key)
    if book:
        return book["title"]
    for k, b in BOOKS.items():
        if b["title"].lower() == book_key.lower():
            return b["title"]
    if (OUTPUT_DIR / book_key).is_dir():
        return book_key
    target_clean = book_key.lower().replace(".pdf", "")
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and (target_clean == d.name.lower() or target_clean in d.name.lower()):
            return d.name
    return None

def verify_book_output(book_key: str) -> bool:
    title = resolve_verify_title(book_key)
    if not title:
        print(f"Error: Unknown book {book_key} and no matching output directory in {OUTPUT_DIR}")
        return False
    book_dir = OUTPUT_DIR / title
    print("=" * 80)
    print(f"VERIFYING DELIVERABLES FOR: {title}")
    print(f"Directory: {book_dir}")
    print("=" * 80)

    all_passed = True

    # 1. Check required subdirectories (ch, vi mandatory; misc/blank optional if no cover/blank pages)
    for sub in ["ch", "vi"]:
        sub_path = book_dir / sub
        if not sub_path.exists():
            print(f"[FAIL] Missing subdirectory: {sub}")
            all_passed = False
        else:
            print(f"[PASS] Subdirectory '{sub}' present with {len(list(sub_path.iterdir()))} items.")

    misc_path = book_dir / "misc"
    if misc_path.exists():
        print(f"[PASS] Subdirectory 'misc' present with {len(list(misc_path.iterdir()))} items.")

    # Check that no temporary or forbidden directories exist
    for unwanted in ["extracted_all", "cache", "tmp"]:
        if (book_dir / unwanted).exists():
            print(f"[FAIL] Unwanted temporary directory found in submission: {unwanted}")
            all_passed = False

    # 2. Check PDF file
    pdf_file = book_dir / f"{title}.pdf"
    if pdf_file.exists():
        print(f"[PASS] Original PDF present: {pdf_file.name} ({pdf_file.stat().st_size} bytes)")
    else:
        print(f"[FAIL] Missing PDF: {pdf_file}")
        all_passed = False

    # 3. Check PPOCRLabel format in ch/ and vi/
    for folder in ["ch", "vi"]:
        label_file = book_dir / folder / "Label.txt"
        state_file = book_dir / folder / "fileState.txt"
        
        if not label_file.exists():
            print(f"[FAIL] Missing {folder}/Label.txt")
            all_passed = False
            continue
        if not state_file.exists():
            print(f"[FAIL] Missing {folder}/fileState.txt")
            all_passed = False
            continue

        # Validate Label.txt (format + real OCR content per guide/04 & 05)
        line_count = 0
        box_errors = 0
        total_boxes = 0
        placeholder_boxes = 0
        han_boxes = 0
        space_in_nom_boxes = 0
        with open(label_file, "r", encoding="utf-8") as f:
            for l_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"  [FAIL] {folder}/Label.txt line {l_idx}: Invalid tab split")
                    box_errors += 1
                    continue
                if not parts[0].startswith(f"{folder}/"):
                    print(f"  [FAIL] {folder}/Label.txt line {l_idx}: Path must start with '{folder}/'")
                    box_errors += 1
                try:
                    boxes = json.loads(parts[1])
                    for b in boxes:
                        total_boxes += 1
                        if len(b.get("points", [])) != 4:
                            print(f"  [FAIL] {folder}/Label.txt line {l_idx}: Box does not have exactly 4 points")
                            box_errors += 1
                        txt = str(b.get("transcription", "") or "")
                        if not txt.strip() or _PLACEHOLDER_RE.search(txt) or txt.startswith(title):
                            placeholder_boxes += 1
                        if folder == "ch":
                            if _HAN_RE.search(txt):
                                han_boxes += 1
                            if " " in txt.strip():
                                space_in_nom_boxes += 1
                except Exception as e:
                    print(f"  [FAIL] {folder}/Label.txt line {l_idx}: JSON error: {e}")
                    box_errors += 1

        if box_errors == 0:
            print(f"[PASS] {folder}/Label.txt format valid ({line_count} images, {total_boxes} boxes, all 4-point).")
        else:
            all_passed = False

        if total_boxes > 0:
            ph_pct = 100.0 * placeholder_boxes / total_boxes
            if ph_pct > 5.0:
                print(f"[FAIL] {folder}/Label.txt has {placeholder_boxes}/{total_boxes} ({ph_pct:.1f}%) empty/placeholder transcriptions!")
                all_passed = False
            else:
                print(f"[PASS] {folder}/Label.txt real transcriptions: {total_boxes - placeholder_boxes}/{total_boxes} ({100.0 - ph_pct:.1f}%).")
            if folder == "ch":
                han_pct = 100.0 * han_boxes / total_boxes
                if han_pct < 90.0:
                    print(f"[FAIL] ch/Label.txt only {han_pct:.1f}% boxes contain real Han-Nom chars!")
                    all_passed = False
                if space_in_nom_boxes > 0:
                    print(f"[WARN] ch/Label.txt has {space_in_nom_boxes} boxes with whitespace inside Nom text (guide 05: no spaces).")

        # Validate fileState.txt
        state_count = 0
        state_errors = 0
        with open(state_file, "r", encoding="utf-8") as f:
            for s_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                state_count += 1
                parts = line.split("\t")
                if len(parts) != 2 or parts[1] not in ["0", "1"]:
                    state_errors += 1
        if state_errors == 0 and state_count == line_count:
            print(f"[PASS] {folder}/fileState.txt valid ({state_count} images with state '1').")
        else:
            print(f"[FAIL] {folder}/fileState.txt has {state_errors} errors (state_count={state_count}, label_count={line_count})")
            all_passed = False

    # 4. Check CSV format + content + similarity metric (>=95% per guide/07)
    csv_file = book_dir / f"{title}.csv"
    if not csv_file.exists():
        print(f"[FAIL] Missing CSV file: {csv_file}")
        all_passed = False
    else:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig", dtype=str)
            expected_cols = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]
            if list(df.columns) == expected_cols:
                print(f"[PASS] CSV header valid ({len(df)} rows, columns: {expected_cols}).")
            else:
                print(f"[FAIL] CSV columns mismatch: {list(df.columns)}")
                all_passed = False

            nom_filled = df["SinoNom_Char"].fillna("").str.strip().ne("").mean() * 100.0 if len(df) else 0.0
            qn_filled = df["ChuQN_txt"].fillna("").str.strip().ne("").mean() * 100.0 if len(df) else 0.0
            if nom_filled < 95.0 or qn_filled < 95.0:
                print(f"[FAIL] CSV content incomplete: SinoNom_Char filled={nom_filled:.1f}%, ChuQN_txt filled={qn_filled:.1f}%")
                all_passed = False
            else:
                print(f"[PASS] CSV content filled: SinoNom_Char={nom_filled:.1f}%, ChuQN_txt={qn_filled:.1f}%.")

            # Evaluate official similarity percentage
            recs = df.fillna("").to_dict("records")
            ev = AlignmentEvaluator().evaluate_dataset(recs)
            sim = ev["similarity_percentage"]
            if ev["total_chars"] == 0 or not ev["is_passed"]:
                print(f"[FAIL] Official similarity = {sim:.2f}% ({ev['total_black_chars']}/{ev['total_chars']} chars, threshold >= {MIN_SIMILARITY_THRESHOLD*100:.0f}%).")
                all_passed = False
            else:
                print(f"[PASS] Official similarity = {sim:.2f}% ({ev['total_black_chars']}/{ev['total_chars']} chars >= 95%).")
        except Exception as e:
            print(f"[FAIL] Error reading CSV: {e}")
            all_passed = False

    # 5. Check XLSX format and character-level / cell-level colors
    xlsx_file = book_dir / f"{title}.xlsx"
    if not xlsx_file.exists():
        print(f"[FAIL] Missing XLSX file: {xlsx_file}")
        all_passed = False
    else:
        try:
            wb = openpyxl.load_workbook(xlsx_file, rich_text=True)
            ws = wb.active
            headers = [ws.cell(1, c).value for c in range(1, 5)]
            expected_cols = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]
            if headers == expected_cols:
                print(f"[PASS] XLSX valid ({ws.max_row - 1} data rows, header matches).")
            else:
                print(f"[FAIL] XLSX header mismatch: {headers}")
                all_passed = False

            black_count = 0
            red_count = 0
            rich_rows = 0
            for r in range(2, min(100, ws.max_row + 1)):
                val = ws.cell(r, 3).value
                if isinstance(val, CellRichText):
                    rich_rows += 1
                    for blk in val:
                        rgb = blk.font.color.rgb if (blk.font and blk.font.color) else COLOR_BLACK
                        if rgb == COLOR_RED:
                            red_count += 1
                        else:
                            black_count += 1
                else:
                    cell_font = ws.cell(r, 3).font
                    if cell_font and cell_font.color and cell_font.color.rgb == COLOR_RED:
                        red_count += 1
                    else:
                        black_count += 1
            print(f"[PASS] XLSX color styling verified (sample: {black_count} black, {red_count} red segments; {rich_rows} rich-text rows).")
        except Exception as e:
            print(f"[FAIL] Error reading XLSX: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    print(f"OVERALL VERIFICATION: {'ALL CHECKS PASSED [SUCCESS]' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 80 + "\n")
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Deliverables Compliance")
    parser.add_argument("--book", type=str, default="AB_160",
                        help="Book key or title to verify (default: AB_160, or 'all')")
    args = parser.parse_args()

    if args.book == "all":
        all_passed = True
        for k in BOOKS.keys():
            if not verify_book_output(k):
                all_passed = False
        sys.exit(0 if all_passed else 1)
    else:
        success = verify_book_output(args.book)
        sys.exit(0 if success else 1)
