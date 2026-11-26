"""Tier C (3 Printed T2b books) & Tier D (12 Remaining Handwritten books) Real OCR Reprocessor.

Strictly compliant with guide/Quy_dinh_do_an.pdf:
  - 300 DPI PDF rendering (no low-res embedded extraction)
  - Clean ink-layer separation before OCR (removes secondary script & notebook gridlines)
  - Page-level KimHanNom Cloud OCR for Hán Nôm (4-worker parallel, resumable JSONL cache)
  - VietOCR (vgg_transformer) for Quốc ngữ line OCR + Dual-Modality Phonetic Fusion
    (cross-checks VietOCR output against NomDict v2 readings & KimHanNom transliteration)
  - Exports compliant PPOCRLabel (ch/, vi/, blank/, misc/, Label.txt, fileState.txt),
    4-column CSV, and character-level Black/Red CellRichText XLSX.

Usage:
  python -u scripts/reprocess_ocr_books.py --tier C --workers 4 --device cpu
  python -u scripts/reprocess_ocr_books.py --book "Tế cô hồn văn SV4.0 Tuệ Phương" --device cpu
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.api_client import KimHanNomClient  # noqa: E402
from core.cleaner import PPOCRLabelFormatter, TextCleaner  # noqa: E402
from core.config import DATASETS_MIDTERM_DIR, OUTPUT_DIR  # noqa: E402
from core.evaluator import AlignmentEvaluator  # noqa: E402
from core.exporter import CorpusExporter  # noqa: E402
from core.nomdict import NomDict, key_base, key_to_text, syl_key  # noqa: E402
from core.preprocess import DocumentPreprocessor, cv2_imread_unicode  # noqa: E402

PDF_DIR = DATASETS_MIDTERM_DIR / "pdf"
CACHE_DIR = ROOT / "data" / "interim"
KHN_PAGE_CACHE = CACHE_DIR / "reprocess_khn_pages.jsonl"
TRANSLIT_CACHE = CACHE_DIR / "reprocess_translit.jsonl"

_HAN_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF\uE000-\uF8FF\U00020000-\U0003FFFF\U000F0000-\U0010FFFF]")
_VI_WORD_RE = re.compile(r"[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+")

# Tier C: 3 printed bilingual books
TIER_C_BOOKS = [
    {"title": "Tế cô hồn văn SV4.0 Tuệ Phương", "mode": "split_lr_h", "layout": "T2b", "is_vertical": False},
    {"title": "Tho Ho Xuan Huong [chu nom]", "mode": "interleaved_h", "layout": "T2b", "is_vertical": False},
    {"title": "ThoNomHoXuanHuongHoangXuanHanSuuTapVaGhiChu_HoXuanHuong", "mode": "split_tb_v", "layout": "T2b", "is_vertical": True},
]

# Tier D: 12 remaining handwritten books (ordered shortest -> longest for steady progress)
TIER_D_BOOKS = [
    {"title": "Thư Thưc Tân Chuyên, Nôm_Y", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Nhị Thập Tứ Trung Diễn Ca (AB.291, copy 1 of 2)_digcoll_37046_Y", "mode": "yale_v", "layout": "T2bw", "is_vertical": True},
    {"title": "TìBàQuốcÂmTânTruyện_琵琶國音新傳_1", "mode": "pure_nom_v", "layout": "T2w", "is_vertical": True},
    {"title": "TìBàQuốcÂmTânTruyện_琵琶國音新傳_2", "mode": "pure_nom_v", "layout": "Tnomw", "is_vertical": True},
    {"title": "Nhi thap tu hieu dien ca digcoll_35504_Y", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Thánh Tô Kê Diên Quôc Âm, Nôm", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_1", "mode": "alternating_v", "layout": "T2w", "is_vertical": True},
    {"title": "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Thâp Nhị Tiên Nàng Thỉnh Luyên Bí Pháp, Nôm", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Thư Vị Văn Châu, Nôm_Y", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Tì Bà Quôc Âm Tân Truyên, Preface in Hán, text in Nôm_Y", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
    {"title": "Tây Dương Quân Thiên Lạc, Nôm, 2 volumes_Y", "mode": "yale_h", "layout": "T2bw", "is_vertical": False},
]

ALL_SPECS = {b["title"]: b for b in TIER_C_BOOKS + TIER_D_BOOKS}


class TokenManager:
    """Thread-safe KimHanNom JWT token manager with periodic & on-error auto-refresh."""

    def __init__(self, refresh_sec: int = 1080):
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._ts: float = 0.0
        self._refresh_sec = refresh_sec
        self.refresh(force=True)

    def get(self) -> str:
        with self._lock:
            if not self._token or (time.time() - self._ts) > self._refresh_sec:
                self._do_login()
            return self._token

    def refresh(self, force: bool = False) -> str:
        with self._lock:
            if force or not self._token or (time.time() - self._ts) > 30:
                self._do_login()
            return self._token

    def _do_login(self):
        for attempt in range(4):
            try:
                cli = KimHanNomClient(auto_login=True)
                self._token = cli.token
                self._ts = time.time()
                return
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3.0 * (attempt + 1))


def load_jsonl_cache(path: Path, key_field: str) -> Dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    # Only treat successful entries as cached so failed API calls retry cleanly
                    if rec.get("ok", True):
                        out[rec[key_field]] = rec
                except Exception:
                    pass
    return out


def append_jsonl(path: Path, rec: dict, lock: threading.Lock):
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def edit_dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_clean_ch_image(img_bgr: np.ndarray, title: str, mode: str, img_path: Path, prep: DocumentPreprocessor) -> np.ndarray:
    """Produce a clean Hán Nôm-only image for page-level KimHanNom OCR."""
    h, w = img_bgr.shape[:2]
    if mode == "split_lr_h":
        # Left half is printed Nôm, right half is printed Quốc ngữ (Te co hon van)
        out = img_bgr.copy()
        out[:, int(w * 0.46):] = 255
        return out
    if mode == "split_tb_v":
        # Top half is vertical Nôm, bottom half is printed QN (ThoNomHoXuanHuong)
        seg = prep.segment_t2b_lines(img_path)
        ch_boxes = seg.get("ch", [])
        if ch_boxes:
            max_y = max(max(pt[1] for pt in b["points"]) for b in ch_boxes)
            cut_y = min(h, int(max_y + 40))
        else:
            cut_y = int(h * 0.52)
        out = img_bgr.copy()
        out[cut_y:, :] = 255
        return out
    if mode == "interleaved_h":
        # Mask out detected vi boxes so KimHanNom only sees Nôm lines
        seg = prep.segment_t2b_lines(img_path)
        out = img_bgr.copy()
        for vb in seg.get("vi", []):
            pts = np.array(vb["points"])
            x0, y0 = np.floor(pts.min(0)).astype(int)
            x1, y1 = np.ceil(pts.max(0)).astype(int)
            out[max(0, y0 - 2):min(h, y1 + 2), max(0, x0 - 2):min(w, x1 + 2)] = 255
        return out
    if mode in ("pure_nom_v", "alternating_v"):
        return img_bgr
    # Yale handwritten T2bw: isolate Nôm ink layer (local Gaussian bg removes yellow paper + gridlines)
    clean = DocumentPreprocessor.synthesize_preprocessed_layer_image(img_bgr, book_title=title, target_script="ch")
    # Mask top-left Yale library catalog stamp ("Yale University Library / Call No.") so it is never OCR'd as Nom
    if mode in ("yale_h", "yale_v") and ("_y" in title.lower() or "thánh" in title.lower() or "thâp nhị" in title.lower()):
        clean[:int(h * 0.11), :int(w * 0.42)] = 255
    return clean


def sort_ocr_boxes(raw_bboxes: list, raw_texts: list, is_vertical: bool, img_w: int) -> List[dict]:
    """Normalize KimHanNom output into 4-point boxes sorted in official reading order."""
    items = []
    for idx, txt in enumerate(raw_texts):
        cleaned = TextCleaner.clean_nom(str(txt or ""))
        han_chars = _HAN_RE.findall(cleaned)
        if not han_chars:
            continue
        if idx >= len(raw_bboxes) or not raw_bboxes[idx]:
            continue
        raw_entry = raw_bboxes[idx]
        poly = raw_entry[0] if (isinstance(raw_entry, list) and len(raw_entry) >= 1 and isinstance(raw_entry[0], list)) else raw_entry
        if not poly or len(poly) != 4:
            continue
        pts = [[int(round(p[0])), int(round(p[1]))] for p in poly]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        items.append({"transcription": "".join(han_chars), "points": pts, "cx": cx, "cy": cy})

    if is_vertical:
        # Check if 2-page spread: left page R->L first, then right page R->L (guide/04)
        left = [it for it in items if it["cx"] < img_w * 0.5]
        right = [it for it in items if it["cx"] >= img_w * 0.5]
        if left and right and len(items) >= 8:
            left.sort(key=lambda it: (-round(it["cx"] / 40.0), it["cy"]))
            right.sort(key=lambda it: (-round(it["cx"] / 40.0), it["cy"]))
            ordered = left + right
        else:
            items.sort(key=lambda it: (-round(it["cx"] / 40.0), it["cy"]))
            ordered = items
    else:
        # Horizontal: group fragments sharing the same y-band into single full verse lines (left -> right)
        items.sort(key=lambda it: (it["cy"], it["cx"]))
        rows = []
        for it in items:
            pts = it["points"]
            iy0 = min(p[1] for p in pts); iy1 = max(p[1] for p in pts)
            ih_b = max(24.0, iy1 - iy0)
            placed = False
            for row in rows:
                rcy = sum(x["cy"] for x in row) / len(row)
                rh = max(24.0, sum(max(p[1] for p in x["points"]) - min(p[1] for p in x["points"]) for x in row) / len(row))
                if abs(it["cy"] - rcy) <= max(ih_b, rh) * 0.58:
                    row.append(it)
                    placed = True
                    break
            if not placed:
                rows.append([it])
        rows.sort(key=lambda row: sum(x["cy"] for x in row) / len(row))
        ordered = []
        for row in rows:
            row.sort(key=lambda x: x["cx"])
            all_xs = [p[0] for x in row for p in x["points"]]
            all_ys = [p[1] for x in row for p in x["points"]]
            merged_txt = "".join(x["transcription"] for x in row)
            # Filter tiny isolated 1-char specks in extreme top/left scanner margin
            if len(merged_txt) == 1 and (min(all_xs) < img_w * 0.08 or min(all_ys) < 120):
                continue
            x0, y0, x1, y1 = min(all_xs), min(all_ys), max(all_xs), max(all_ys)
            ordered.append({"transcription": merged_txt, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})

    return [{"transcription": it["transcription"], "points": it["points"]} for it in ordered]


def fuse_nom_and_vi(
    nom_text: str,
    vi_raw: str,
    nd: NomDict,
    khn_translit_text: str = "",
) -> str:
    """Dual-Modality Phonetic Fusion:
    Aligns each Hán Nôm character's NomDict v2 readings against VietOCR visual prediction
    (and KimHanNom transliteration fallback) to repair OCR slips while ensuring phonetic fidelity.
    """
    nom_chars = _HAN_RE.findall(nom_text)
    if not nom_chars:
        return TextCleaner.clean_quocngu(vi_raw or khn_translit_text)

    vi_words = _VI_WORD_RE.findall(vi_raw or "")
    tr_words = _VI_WORD_RE.findall(khn_translit_text or "")

    # If VietOCR returned clean matching word count, refine each word with NomDict v2
    if len(vi_words) == len(nom_chars):
        out = []
        for idx, (ch, vw) in enumerate(zip(nom_chars, vi_words)):
            vw_low = vw.lower()
            if nd.match(ch, vw_low):
                out.append(vw_low)
                continue
            readings = nd.c2k.get(ch, set())
            if readings:
                vw_key = syl_key(vw_low)
                vw_base = key_base(vw_key)
                # Prefer reading with smallest edit distance to VietOCR visual prediction
                best_k = min(
                    readings,
                    key=lambda k: (edit_dist(key_base(k), vw_base), k[-1] != vw_key[-1]),
                )
                d_base = edit_dist(key_base(best_k), vw_base)
                if d_base <= 2:
                    out.append(key_to_text(best_k))
                    continue
                # If KimHanNom transliteration matches a reading of ch, trust it over noisy handwriting OCR
                if idx < len(tr_words) and nd.match(ch, tr_words[idx].lower()):
                    out.append(tr_words[idx].lower())
                    continue
            out.append(vw_low)
        return TextCleaner.clean_quocngu(" ".join(out))

    # If lengths differ, use KimHanNom transliteration when its length matches nom_chars
    if len(tr_words) == len(nom_chars):
        out = []
        vi_low_set = [w.lower() for w in vi_words]
        for ch, tw in zip(nom_chars, tr_words):
            tw_low = tw.lower()
            if nd.match(ch, tw_low):
                out.append(tw_low)
                continue
            readings = nd.c2k.get(ch, set())
            # Check if any word in VietOCR matches a reading of ch
            matched_vi = next((w for w in vi_low_set if nd.match(ch, w)), None)
            if matched_vi:
                out.append(matched_vi)
            elif readings:
                best_k = min(readings, key=lambda k: edit_dist(key_base(k), key_base(syl_key(tw_low))))
                out.append(key_to_text(best_k))
            else:
                out.append(tw_low)
        return TextCleaner.clean_quocngu(" ".join(out))

    # Fallback: pick closest reading per char guided by vi_words / tr_words
    if vi_words and len(vi_words) >= max(2, int(0.6 * len(nom_chars))):
        return TextCleaner.clean_quocngu(" ".join(vi_words))
    if tr_words:
        return TextCleaner.clean_quocngu(" ".join(tr_words))
    out = []
    for ch in nom_chars:
        rs = nd.c2k.get(ch, set())
        out.append(key_to_text(sorted(rs)[0]) if rs else "?")
    return TextCleaner.clean_quocngu(" ".join(out))


class VietOCRRunner:
    def __init__(self, device: str = "cpu"):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["cnn"]["pretrained"] = False
        cfg["device"] = device
        self.det = Predictor(cfg)

    def predict_crop(self, bgr: np.ndarray) -> str:
        if bgr is None or bgr.size == 0 or bgr.shape[0] < 8 or bgr.shape[1] < 8:
            return ""
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        try:
            return str(self.det.predict(pil) or "").strip()
        except Exception:
            return ""


def process_book(spec: dict, khn_cache: dict, tr_cache: dict, khn_lock: threading.Lock, tr_lock: threading.Lock,
                 nd: NomDict, vi_ocr: VietOCRRunner, token_mgr: TokenManager, workers: int = 2):
    title = spec["title"]
    mode = spec["mode"]
    layout = spec["layout"]
    is_vertical = spec["is_vertical"]
    ocr_id = 1 if is_vertical else 4
    read_dir = 1 if is_vertical else 2

    pdf_path = PDF_DIR / f"{title}.pdf"
    if not pdf_path.exists():
        print(f"[SKIP] PDF not found: {pdf_path}")
        return

    print("\n" + "=" * 84)
    print(f"REPROCESSING [{layout} | {mode}]: {title}")
    print("=" * 84)

    book_out_dir = OUTPUT_DIR / title
    for sub in ("ch", "vi", "misc", "blank", "extracted_all"):
        sd = book_out_dir / sub
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)

    prep = DocumentPreprocessor({"title": title, "layout_type": layout}, OUTPUT_DIR)
    extracted = prep.split_and_extract_images(str(pdf_path))
    total_pages = len(extracted)
    print(f"  Rendered {total_pages} pages at 300 DPI.")

    # Step 1: Classify content vs blank vs misc
    content_pages = []
    n_blank = n_misc = 0
    for page_num, img_name, img_path in extracted:
        # Special rule: Hoang Xuan Han vol. 2 pages >= 195 are historical maps -> misc/
        if title == "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2" and page_num >= 195:
            shutil.copy2(str(img_path), str(prep.dir_misc / img_name))
            n_misc += 1
            continue
        pcls = DocumentPreprocessor.classify_page(img_path, page_num, total_pages)
        if pcls == "blank":
            shutil.copy2(str(img_path), str(prep.dir_blank / img_name))
            n_blank += 1
        elif pcls == "misc":
            shutil.copy2(str(img_path), str(prep.dir_misc / img_name))
            n_misc += 1
        else:
            content_pages.append((page_num, img_name, img_path))
    print(f"  Classified: content={len(content_pages)}, blank={n_blank}, misc={n_misc}")

    # Step 2: Parallel page-level KimHanNom OCR on cleaned Nôm layer (cached!)
    todo_pages = []
    for page_num, img_name, img_path in content_pages:
        ckey = f"{title}|{page_num}|{ocr_id}"
        if ckey not in khn_cache:
            todo_pages.append((page_num, img_name, img_path, ckey))

    print(f"  KimHanNom page OCR: total={len(content_pages)} cached={len(content_pages)-len(todo_pages)} todo={len(todo_pages)}")

    def ocr_one_page(item):
        page_num, img_name, img_path, ckey = item
        img = cv2_imread_unicode(img_path)
        clean_ch = build_clean_ch_image(img, title, mode, img_path, prep)
        ch_h, ch_w = clean_ch.shape[:2]
        max_side = max(ch_h, ch_w)

        # Adaptive max-side per retry (smaller image if server 504s!)
        cap_schedule = [1800, 1600, 1400, 1200, 1000]
        backoff = [3.0, 6.0, 12.0, 20.0, 25.0]
        tmp = None
        for attempt in range(5):
            cap = cap_schedule[attempt]
            up_scale = float(cap) / max_side if max_side > cap else 1.0
            upload_img = cv2.resize(clean_ch, (int(ch_w * up_scale), int(ch_h * up_scale)), interpolation=cv2.INTER_AREA) if up_scale < 1.0 else clean_ch

            tok = token_mgr.get()
            cli = KimHanNomClient(auto_login=False)
            cli.token = tok
            cli.session.headers.update({"Authorization": f"Bearer {tok}"})
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    tmp = tf.name
                cv2.imencode(".jpg", upload_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(tmp)
                srv = cli.upload_image(tmp)
                res = cli.ocr_image(srv, ocr_id=ocr_id, lang_type=2, reading_direction=read_dir)
                raw_bboxes = res.get("result_bbox", [])
                if up_scale < 1.0 and raw_bboxes:
                    inv = 1.0 / up_scale
                    scaled_bboxes = []
                    for entry in raw_bboxes:
                        if isinstance(entry, list) and len(entry) >= 1 and isinstance(entry[0], list):
                            new_poly = [[p[0] * inv, p[1] * inv] for p in entry[0]]
                            scaled_bboxes.append([new_poly] + entry[1:])
                        else:
                            scaled_bboxes.append(entry)
                    raw_bboxes = scaled_bboxes
                rec = {
                    "key": ckey,
                    "title": title,
                    "page": page_num,
                    "result_ocr_text": res.get("result_ocr_text", []),
                    "result_bbox": raw_bboxes,
                    "ok": True,
                }
                append_jsonl(KHN_PAGE_CACHE, rec, khn_lock)
                khn_cache[ckey] = rec
                return True
            except Exception as e:
                err_s = str(e)
                if attempt < 4:
                    if any(code in err_s for code in ("401", "403", "token", "auth", "000104")):
                        try:
                            token_mgr.refresh(force=False)
                        except Exception:
                            pass
                    time.sleep(backoff[attempt])
                else:
                    rec = {"key": ckey, "title": title, "page": page_num, "result_ocr_text": [], "result_bbox": [], "ok": False, "error": err_s}
                    append_jsonl(KHN_PAGE_CACHE, rec, khn_lock)
                    khn_cache[ckey] = rec
                    return False
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        return False

    if todo_pages:
        t0 = time.time()
        done_ok = done_err = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(ocr_one_page, it) for it in todo_pages]
            for idx, fut in enumerate(as_completed(futs), 1):
                if fut.result():
                    done_ok += 1
                else:
                    done_err += 1
                if idx % 10 == 0 or idx == len(todo_pages):
                    rate = idx / max(time.time() - t0, 0.1)
                    eta = (len(todo_pages) - idx) / max(rate, 1e-6)
                    print(f"    KHN [{idx:3d}/{len(todo_pages)}] ok={done_ok} err={done_err} ({rate:.2f} p/s, ETA {eta:.0f}s)")

    # Batch-transliterate all unique uncached Nôm lines for this book (25 lines per API call -> 20x faster!)
    all_page_ch_boxes = {}
    unique_nom_todo = []
    seen_nom = set()
    for page_num, img_name, img_path in content_pages:
        ckey = f"{title}|{page_num}|{ocr_id}"
        krec = khn_cache.get(ckey, {})
        img = cv2_imread_unicode(img_path)
        ih, iw = img.shape[:2]
        cboxes = sort_ocr_boxes(krec.get("result_bbox", []), krec.get("result_ocr_text", []), is_vertical, iw)
        all_page_ch_boxes[img_name] = cboxes
        for cb in cboxes:
            nt = cb["transcription"]
            if nt and nt not in tr_cache and nt not in seen_nom:
                seen_nom.add(nt)
                unique_nom_todo.append(nt)

    def translit_batch(lines_in: List[str]):
        tok = token_mgr.get()
        tr_cli = KimHanNomClient(auto_login=False)
        tr_cli.token = tok
        tr_cli.session.headers.update({"Authorization": f"Bearer {tok}"})
        for attempt in range(3):
            try:
                res_list = tr_cli.transliterate("\n".join(lines_in), lang_type=2)
                if len(res_list) == len(lines_in):
                    return [str(q).strip() for q in res_list]
                return None
            except Exception as e:
                if attempt < 2:
                    token_mgr.refresh(force=False)
                    tok = token_mgr.get()
                    tr_cli.token = tok
                    tr_cli.session.headers.update({"Authorization": f"Bearer {tok}"})
                    time.sleep(2.0 * (attempt + 1))
        return None

    if unique_nom_todo:
        print(f"  Batch transliterating {len(unique_nom_todo)} unique Nôm lines (25/batch)...")
        bsz = 25
        for bi in range(0, len(unique_nom_todo), bsz):
            batch = unique_nom_todo[bi:bi + bsz]
            got = translit_batch(batch)
            if got is not None:
                for nt, qn in zip(batch, got):
                    rec = {"nom": nt, "qn": qn}
                    append_jsonl(TRANSLIT_CACHE, rec, tr_lock)
                    tr_cache[nt] = rec
                continue
            # Sub-batch of 5 before 1-by-1
            for si in range(0, len(batch), 5):
                sub = batch[si:si + 5]
                got_s = translit_batch(sub)
                if got_s is not None:
                    for nt, qn in zip(sub, got_s):
                        rec = {"nom": nt, "qn": qn}
                        append_jsonl(TRANSLIT_CACHE, rec, tr_lock)
                        tr_cache[nt] = rec
                    continue
                for nt in sub:
                    got_1 = translit_batch([nt])
                    qn = got_1[0] if got_1 else ""
                    rec = {"nom": nt, "qn": qn}
                    append_jsonl(TRANSLIT_CACHE, rec, tr_lock)
                    tr_cache[nt] = rec

    def get_translit(nom_line: str) -> str:
        return tr_cache.get(nom_line, {}).get("qn", "")

    # Step 3: Build ch/Label.txt, vi/Label.txt, CSV, XLSX
    ch_labels = {}
    vi_labels = {}
    aligned_records = []

    for page_num, img_name, img_path in content_pages:
        img = cv2_imread_unicode(img_path)
        ih, iw = img.shape[:2]
        ch_boxes = all_page_ch_boxes.get(img_name, [])

        # For alternating_v (ThoCaDuHanh_1): some pages are pure QN typed pages!
        if mode == "alternating_v" and len(ch_boxes) == 0:
            # Treat as QN-only page
            shutil.copy2(str(img_path), str(prep.dir_vi / img_name))
            seg = prep.segment_t2b_lines(img_path)
            vboxes = seg.get("vi", []) or seg.get("ch", [])
            vi_page = []
            for vb in vboxes:
                pts = np.array(vb["points"])
                x0, y0 = np.floor(pts.min(0)).astype(int); x1, y1 = np.ceil(pts.max(0)).astype(int)
                txt = TextCleaner.clean_quocngu(vi_ocr.predict_crop(img[max(0, y0 - 4):min(ih, y1 + 4), max(0, x0 - 4):min(iw, x1 + 4)]))
                if txt:
                    vi_page.append({"transcription": txt, "points": vb["points"]})
            if vi_page:
                vi_labels[img_name] = vi_page
            else:
                (prep.dir_vi / img_name).unlink(missing_ok=True)
                shutil.copy2(str(img_path), str(prep.dir_misc / img_name))
            continue

        if not ch_boxes:
            # No text detected -> move to misc/blank
            shutil.copy2(str(img_path), str(prep.dir_misc / img_name))
            n_misc += 1
            continue

        # Save image into ch/
        shutil.copy2(str(img_path), str(prep.dir_ch / img_name))
        ch_labels[img_name] = ch_boxes

        # Detect & OCR Quốc ngữ lines if this page contains QN
        has_onpage_vi = mode in ("split_lr_h", "interleaved_h", "split_tb_v", "yale_h", "yale_v")
        vi_page_boxes = []
        vi_texts_for_page = []

        if has_onpage_vi:
            shutil.copy2(str(img_path), str(prep.dir_vi / img_name))
            seg = prep.segment_t2b_lines(img_path)
            seg_vi = seg.get("vi", [])
            # Prepare clean image for VietOCR
            if mode in ("yale_h", "yale_v"):
                vi_src_img = DocumentPreprocessor.synthesize_preprocessed_layer_image(img, book_title=title, target_script="vi")
            else:
                vi_src_img = img

            # Pair each ch_box with a corresponding vi_box
            for idx_b, cb in enumerate(ch_boxes):
                nom_str = cb["transcription"]
                cpts = np.array(cb["points"])
                cx0, cy0 = cpts.min(0); cx1, cy1 = cpts.max(0)

                if mode == "split_lr_h":
                    # Right column at matching y-band
                    matched_vb = None; best_d = 1e9
                    for vb in seg_vi:
                        vp = np.array(vb["points"])
                        if vp[:, 0].mean() < iw * 0.45:
                            continue
                        d = abs(vp[:, 1].mean() - (cy0 + cy1) / 2.0)
                        if d < best_d:
                            best_d = d; matched_vb = vb["points"]
                    if matched_vb is None or best_d > 120:
                        vx0, vx1 = int(iw * 0.48), int(iw * 0.96)
                        vy0, vy1 = max(0, int(cy0 - 6)), min(ih - 1, int(cy1 + 6))
                        matched_vb = [[vx0, vy0], [vx1, vy0], [vx1, vy1], [vx0, vy1]]
                elif mode == "split_tb_v":
                    # Bottom printed QN lines matched in order
                    if idx_b < len(seg_vi):
                        matched_vb = seg_vi[idx_b]["points"]
                    else:
                        matched_vb = None
                elif mode == "yale_v":
                    # Vertical Nôm column with red Quốc ngữ written immediately beside it (left/right strip)
                    cw = max(35, cx1 - cx0)
                    vx0 = max(0, int(cx1 + 2))
                    vx1 = min(iw - 1, int(cx1 + cw * 1.1))
                    if vx1 - vx0 < 25:
                        vx0 = max(0, int(cx0 - cw * 1.1))
                        vx1 = max(vx0 + 25, int(cx0 - 2))
                    vy0, vy1 = max(0, int(cy0)), min(ih - 1, int(cy1))
                    matched_vb = [[vx0, vy0], [vx1, vy0], [vx1, vy1], [vx0, vy1]]
                else:
                    # Horizontal interleaved / Yale T2bw: vi line sits just below ch line
                    nxt = ch_boxes[idx_b + 1]["points"] if idx_b + 1 < len(ch_boxes) else None
                    y_lim = int(np.array(nxt).min(0)[1]) if nxt is not None else min(ih - 1, int(cy1 + (cy1 - cy0) * 1.4))
                    matched_vb = None; best_d = 1e9
                    for vb in seg_vi:
                        vp = np.array(vb["points"])
                        vyc = vp[:, 1].mean()
                        if vyc > cy0 + (cy1 - cy0) * 0.4 and vyc < y_lim + 20:
                            d = abs(vyc - (cy1 + y_lim) / 2.0)
                            if d < best_d:
                                best_d = d; matched_vb = vb["points"]
                    if matched_vb is None:
                        lh = max(30, cy1 - cy0)
                        vy0 = min(ih - 12, int(cy1 + 2))
                        vy1 = min(ih - 1, max(vy0 + 28, int(min(y_lim - 4, cy1 + lh * 0.95))))
                        vx0, vx1 = max(0, int(cx0 - 10)), min(iw - 1, int(cx1 + 10))
                        matched_vb = [[vx0, vy0], [vx1, vy0], [vx1, vy1], [vx0, vy1]]

                vi_raw = ""
                if matched_vb is not None:
                    vp = np.array(matched_vb)
                    x0, y0 = np.floor(vp.min(0)).astype(int); x1, y1 = np.ceil(vp.max(0)).astype(int)
                    vi_raw = vi_ocr.predict_crop(vi_src_img[max(0, y0 - 4):min(ih, y1 + 4), max(0, x0 - 4):min(iw, x1 + 4)])

                tr_qn = get_translit(nom_str)
                final_vi = fuse_nom_and_vi(nom_str, vi_raw, nd, tr_qn)
                vi_texts_for_page.append(final_vi)
                if matched_vb is not None:
                    vi_page_boxes.append({"transcription": final_vi, "points": [[int(round(p[0])), int(round(p[1]))] for p in matched_vb]})

            vi_labels[img_name] = vi_page_boxes
        else:
            # Pure Nôm page: transliterate via KimHanNom + NomDict v2
            for cb in ch_boxes:
                tr_qn = get_translit(cb["transcription"])
                vi_texts_for_page.append(fuse_nom_and_vi(cb["transcription"], "", nd, tr_qn))

        stem_no_jpg = Path(img_name).stem
        for s_idx, (cb, vtxt) in enumerate(zip(ch_boxes, vi_texts_for_page), start=1):
            aligned_records.append({
                "img_id": img_name,
                "sentence_id": f"{stem_no_jpg}.{s_idx}",
                "SinoNom_Char": cb["transcription"],
                "ChuQN_txt": vtxt,
            })

    prep.cleanup_cache()
    shutil.copy2(str(pdf_path), str(book_out_dir / f"{title}.pdf"))

    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_ch, "ch", ch_labels)
    PPOCRLabelFormatter.write_ppocr_dataset(prep.dir_vi, "vi", vi_labels)

    ev = AlignmentEvaluator().evaluate_dataset(aligned_records)
    CorpusExporter.export_csv(aligned_records, book_out_dir / f"{title}.csv")
    CorpusExporter.export_xlsx(aligned_records, ev["row_evaluations"], book_out_dir / f"{title}.xlsx")
    print(f"  DONE {title}: ch_pages={len(ch_labels)} vi_pages={len(vi_labels)} rows={len(aligned_records)}")
    print(f"  Official Similarity: {ev['similarity_percentage']:.2f}% ({ev['total_black_chars']}/{ev['total_chars']} chars, PASSED={ev['is_passed']})")


def book_already_passed(title: str) -> bool:
    """Return True if output/<title> already has real OCR content and >=95% similarity."""
    bdir = OUTPUT_DIR / title
    csv_f = bdir / f"{title}.csv"
    if not csv_f.exists():
        return False
    try:
        df = pd.read_csv(csv_f, encoding="utf-8-sig", dtype=str).fillna("")
        if len(df) == 0:
            return False
        if df["SinoNom_Char"].str.strip().ne("").mean() < 0.95 or df["ChuQN_txt"].str.strip().ne("").mean() < 0.95:
            return False
        ev = AlignmentEvaluator().evaluate_dataset(df.to_dict("records"))
        return bool(ev["is_passed"] and ev["total_chars"] > 0)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["C", "D", "all"], default="C")
    ap.add_argument("--book", type=str, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-passed", action="store_true")
    args = ap.parse_args()

    nd = NomDict.load()
    khn_cache = load_jsonl_cache(KHN_PAGE_CACHE, "key")
    tr_cache = load_jsonl_cache(TRANSLIT_CACHE, "nom")
    khn_lock = threading.Lock()
    tr_lock = threading.Lock()

    token_mgr = TokenManager(refresh_sec=1080)
    vi_ocr = VietOCRRunner(device=args.device)

    if args.book:
        specs = [ALL_SPECS[args.book]]
    elif args.tier == "C":
        specs = TIER_C_BOOKS
    elif args.tier == "D":
        specs = TIER_D_BOOKS
    else:
        specs = TIER_C_BOOKS + TIER_D_BOOKS

    from scripts.verify_output import verify_book_output

    for idx_s, spec in enumerate(specs, 1):
        title = spec["title"]
        if args.skip_passed and book_already_passed(title):
            print(f"[SKIP-PASSED {idx_s:02d}/{len(specs)}] {title} already passes >=95% with real content.")
            continue
        process_book(spec, khn_cache, tr_cache, khn_lock, tr_lock, nd, vi_ocr, token_mgr, workers=args.workers)
        verify_book_output(title)


if __name__ == "__main__":
    main()
