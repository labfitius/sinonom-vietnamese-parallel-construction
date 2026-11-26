"""Rebuild the printed (non-handwritten) books from real OCR of the page images.

Books (see guide/Quy_định_đồ_án.pdf, "Chú thích về phân loại sách"):
  - Tế cô hồn văn SV4.0 Tuệ Phương   (T2b: table "verse no. | Nôm | Quốc ngữ", typeset)
  - Tho Ho Xuan Huong [chu nom]      (T2b: Nôm line followed by its Quốc ngữ line, typeset)
  - ThuDạLữHoàiNgâm_秋夜旅懷吟        (Tnom: woodblock, vertical columns, 2 pages per image)
  - Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch    (Tqn: Hán-Việt reading of the woodblock, PDF with a text layer)

Sources of each field (nothing is generated from the other script):
  - SinoNom_Char : KimHanNom OCR of the Nôm/Hán region of the page.
  - ChuQN_txt    : VietOCR on the Quốc ngữ line, majority vote over 8 OCR variants
                   (2 models x 4 preprocessings). The Nôm dictionary is only used to break a tie
                   between words that the OCR variants actually read; it never creates a word.
                   For the _Dịch book the text layer of the PDF is used directly.

Usage:
  python -u scripts/rebuild_printed_books.py --book all
  python -u scripts/rebuild_printed_books.py --book te_co_hon
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.api_client import KimHanNomClient  # noqa: E402
from core.cleaner import PPOCRLabelFormatter, TextCleaner  # noqa: E402
from core.config import OUTPUT_DIR  # noqa: E402
from core.dictionary import ITERATION_MARKS  # noqa: E402
from core.evaluator import AlignmentEvaluator  # noqa: E402
from core.exporter import CorpusExporter  # noqa: E402
from core.nomdict import NomDict, VI_WORD, key_to_text  # noqa: E402
from core.preprocess import cv2_imread_unicode  # noqa: E402

KHN_CACHE = ROOT / "data" / "interim" / "printed_khn_cache.jsonl"
HAN_RE = re.compile(r"[㐀-鿿豈-﫿-\U00020000-\U0003FFFF\U000F0000-\U0010FFFF]")
RENDER_DPI = 300

TE_CO_HON = "Tế cô hồn văn SV4.0 Tuệ Phương"
THO_HXH = "Tho Ho Xuan Huong [chu nom]"
THU_DA = "ThuDạLữHoàiNgâm_秋夜旅懷吟"
THU_DA_DICH = "Thụ Dạ Lữ Hoài Ngâm Khúc_Dịch"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def imwrite_unicode(path: Path, img: np.ndarray):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(path))


def runs(mask: np.ndarray, min_len: int = 1, max_gap: int = 0) -> List[Tuple[int, int]]:
    """[start, end) runs of True in a 1D mask; gaps <= max_gap are merged."""
    out: List[List[int]] = []
    idx = np.flatnonzero(mask)
    for i in idx:
        if out and i - out[-1][1] <= max_gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [(a, b) for a, b in out if b - a >= min_len]


def ink_mask(img: np.ndarray, thr: int = 170) -> np.ndarray:
    """Ink pixels after dividing out the (textured / yellowed) paper background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return (norm < thr).astype(np.uint8)


def flatten(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    return cv2.cvtColor(cv2.divide(gray, np.maximum(bg, 1), scale=255), cv2.COLOR_GRAY2BGR)


def binarize(img: np.ndarray) -> np.ndarray:
    norm = flatten(img)[:, :, 0]
    return cv2.cvtColor(np.where(norm < 175, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def tight_box(ink: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 6) -> Optional[List[List[int]]]:
    """Shrink a region to its ink extent; returns 4 points or None if empty."""
    sub = ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    h, w = ink.shape
    a, b = int(max(0, x0 + xs.min() - pad)), int(min(w - 1, x0 + xs.max() + pad))
    c, d = int(max(0, y0 + ys.min() - pad)), int(min(h - 1, y0 + ys.max() + pad))
    return [[a, c], [b, c], [b, d], [a, d]]


def render_pdf(pdf_path: Path, out_dir: Path, title: str) -> List[Tuple[int, str, Path]]:
    """Render every PDF page to <title>.<NNN>.jpg (numbering from 1, >=3 digits)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, pg in enumerate(doc, start=1):
        name = f"{title}.{i:03d}.jpg" if len(doc) <= 999 else f"{title}.{i}.jpg"
        p = out_dir / name
        if not p.exists():
            pix = pg.get_pixmap(dpi=RENDER_DPI)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            imwrite_unicode(p, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        pages.append((i, name, p))
    return pages


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------
class KHN:
    """KimHanNom OCR with a resumable JSONL cache keyed by (book, image, region, mode)."""

    def __init__(self):
        self.cache: Dict[str, dict] = {}
        if KHN_CACHE.exists():
            for line in KHN_CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.cli: Optional[KimHanNomClient] = None

    offline = False  # --offline: use cached results only (e.g. while the API refuses access)
    MIN_INTERVAL = 2.5  # seconds between API calls: bursts get the account refused (HTTP 000403)
    _last_call = 0.0

    def ocr(self, key: str, img: np.ndarray, vertical: bool, lang_type: int = 2) -> List[Tuple[str, np.ndarray, float]]:
        """Returns [(text, 4x2 points in img coords, confidence)]."""
        if key not in self.cache and self.offline:
            return []
        if key not in self.cache:
            if self.cli is None:
                self.cli = KimHanNomClient(auto_login=True)
            tmp = Path(tempfile.gettempdir()) / "khn_region.jpg"
            imwrite_unicode(tmp, img)
            last_err = None
            for attempt in range(6):
                time.sleep(max(0.0, KHN._last_call + self.MIN_INTERVAL - time.time()))
                KHN._last_call = time.time()
                try:
                    res = self.cli.ocr_image(self.cli.upload_image(str(tmp)), ocr_id=1 if vertical else 4,
                                             lang_type=lang_type, reading_direction=1 if vertical else 2)
                    break
                except Exception as e:  # token expiry / transient network or login errors
                    last_err = e
                    time.sleep(5 * (attempt + 1))
                    try:
                        self.cli = KimHanNomClient(auto_login=True)
                    except Exception as e2:
                        last_err = e2
            else:
                raise RuntimeError(f"KimHanNom failed for {key}: {last_err}")
            rec = {"key": key, "items": [[t, b[0], b[1][1]] for t, b in zip(res["result_ocr_text"], res["result_bbox"])]}
            with open(KHN_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[key] = rec
        return [(t, np.array(p, dtype=float), float(c)) for t, p, c in self.cache[key]["items"]]


class QNVoter:
    """VietOCR ensemble: 2 models x 4 preprocessings, word-level majority vote."""

    VARIANTS = (
        ("raw", lambda c: c),
        ("flat", flatten),
        ("bin", binarize),
        ("half", lambda c: cv2.resize(c, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
    )

    def __init__(self, nd: NomDict, device: str = "cuda"):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        self.nd = nd
        self.models = []
        for name in ("vgg_transformer", "vgg_seq2seq"):
            cfg = Cfg.load_config_from_name(name)
            cfg["device"] = device
            self.models.append(Predictor(cfg))

    def read_all(self, crop: np.ndarray) -> List[str]:
        outs = []
        for m in self.models:
            for _, fn in self.VARIANTS:
                pil = Image.fromarray(cv2.cvtColor(fn(crop), cv2.COLOR_BGR2RGB))
                outs.append(str(m.predict(pil) or "").strip())
        return outs

    def vote(self, readings: List[str], nom_chars: List[str]) -> Tuple[str, dict]:
        """Majority vote per word position among variants that agree on the word count.

        The Nôm character only breaks a tie between candidates proposed by the OCR variants.
        """
        toks = [r.split() for r in readings if r.strip()]
        if not toks:
            return "", {"n_variants": 0}
        counts = Counter(len(t) for t in toks)
        # Prefer the word count equal to the Nôm length if at least 2 variants produced it
        n_words = len(nom_chars) if counts.get(len(nom_chars), 0) >= 2 else counts.most_common(1)[0][0]
        pool = [t for t in toks if len(t) == n_words]
        out, n_tie = [], 0
        for pos in range(n_words):
            c = Counter(t[pos] for t in pool)
            top = c.most_common()
            best_n = top[0][1]
            tied = [w for w, n in top if n == best_n]
            pick = tied[0]
            if len(tied) > 1 and pos < len(nom_chars) and len(nom_chars) == n_words:
                ok = [w for w in tied if self.nd.match(nom_chars[pos], re.sub(r"[^\w]", "", w).lower())]
                if ok:
                    pick = ok[0]
                    n_tie += 1
            out.append(pick)
        return " ".join(out), {"n_variants": len(pool), "dict_tiebreaks": n_tie}


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------
def write_book(title: str, pages: List[Tuple[int, str, Path]], folders: Dict[str, List[str]],
               ch_labels: Dict[str, list], vi_labels: Dict[str, list], records: List[dict], pdf_src: Path):
    """Write ch/ vi/ blank/ misc/ + Label.txt/fileState.txt + CSV/XLSX for one book."""
    book_dir = OUTPUT_DIR / title
    for sub in ("ch", "vi", "blank", "misc", "intro", "ref", "toc"):
        if (book_dir / sub).exists():
            shutil.rmtree(book_dir / sub)
    by_name = {name: p for _, name, p in pages}
    for sub, names in folders.items():
        (book_dir / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(by_name[n], book_dir / sub / n)
    for sub, labels in (("ch", ch_labels), ("vi", vi_labels)):
        (book_dir / sub).mkdir(parents=True, exist_ok=True)
        missing = set(folders.get(sub, [])) - set(labels)
        assert not missing, f"{sub}: images without labels {sorted(missing)[:3]}"
        PPOCRLabelFormatter.write_ppocr_dataset(book_dir / sub, sub, labels)
    if pdf_src.resolve() != (book_dir / f"{title}.pdf").resolve():
        shutil.copy2(pdf_src, book_dir / f"{title}.pdf")
    ev = AlignmentEvaluator().evaluate_dataset(records)
    CorpusExporter.export_csv(records, book_dir / f"{title}.csv")
    CorpusExporter.export_xlsx(records, ev["row_evaluations"], book_dir / f"{title}.xlsx")
    print(f"  {title}: rows={len(records)} similarity={ev['similarity_percentage']:.2f}% "
          f"({ev['total_black_chars']}/{ev['total_chars']})")
    return ev


def nom_line(text: str) -> str:
    return TextCleaner.clean_nom("".join(HAN_RE.findall(text)))


def char_cells(ink_line: np.ndarray, vertical: bool = False) -> List[Tuple[int, int]]:
    """Split a typeset Nôm line into character cells along its reading axis.

    Printed glyphs sit on a regular pitch, so runs that are too narrow are merged (left/right
    components of one glyph) and runs spanning k pitches are split into k cells.
    """
    if vertical:
        ink_line = ink_line.T
    H = ink_line.shape[0]
    r = runs(ink_line.sum(0) > 0, min_len=1, max_gap=max(2, int(H * 0.06)))
    merged: List[List[int]] = []
    for a, b in r:
        if merged and (b - merged[-1][0]) <= 1.15 * H and ((b - a) < 0.55 * H or (merged[-1][1] - merged[-1][0]) < 0.55 * H):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    widths = [b - a for a, b in merged if 0.6 * H <= b - a <= 1.2 * H]
    pitch = float(np.median(widths)) if widths else H * 0.95
    out = []
    for a, b in merged:
        k = max(1, int(round((b - a) / pitch)))
        out += [(a + (b - a) * i // k, a + (b - a) * (i + 1) // k) for i in range(k)]
    return out


def white_pad(img: np.ndarray, pad: int) -> np.ndarray:
    return cv2.copyMakeBorder(flatten(img), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def read_nom_line(khn: KHN, key: str, img: np.ndarray, ink: np.ndarray, box: List[List[int]],
                  column_text: str, vertical: bool = False) -> Tuple[str, dict]:
    """OCR one typeset Nôm line so that it has exactly one character per glyph cell.

    Voting between KimHanNom runs (whole column, the single line, the line at 0.6x) picks the
    reading whose length equals the counted glyph cells; if none does, cells that the line OCR
    skipped (rare glyphs are silently dropped) are filled by OCR of that single cell.
    """
    p = np.array(box)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    cells = char_cells(ink[y0:y1, x0:x1], vertical)
    n = len(cells)
    line = img[y0:y1, x0:x1]
    cands = [("column", column_text)]
    for tag, crop in (("line", white_pad(line, 150)),
                      ("line0.6", cv2.resize(white_pad(line, 150), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))):
        if any(len(t) == n for _, t in cands):
            break
        items = khn.ocr(f"{key}|{tag}", crop, vertical=vertical)
        order = (lambda k: k[1][:, 1].min()) if vertical else (lambda k: k[1][:, 0].min())
        cands.append((tag, nom_line("".join(t for t, _, _ in sorted(items, key=order)))))
    for tag, text in cands:
        if len(text) == n:
            return text, {"nom_source": tag, "cells": n}
    # Fill from single-cell OCR, keeping the line OCR wherever the two agree in order.
    per_cell = []
    for i, (a, b) in enumerate(cells):
        cell = line[:, a:b] if not vertical else line[a:b, :]
        items = khn.ocr(f"{key}|cell{i}", white_pad(cell, 60), vertical=False)
        per_cell.append(nom_line("".join(t for t, _, _ in items))[:1])
    import difflib
    best = max(cands, key=lambda c: difflib.SequenceMatcher(None, c[1], "".join(per_cell)).ratio())[1]
    out: List[str] = []
    sm = difflib.SequenceMatcher(None, list(best), per_cell, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            out += list(best[i1:i2])
        elif op in ("insert", "replace"):
            out += per_cell[j1:j2]
    text = "".join(out)
    return text, {"nom_source": "cells", "cells": n, "per_cell": "".join(per_cell), "line_best": best}


def strip_marks(s: str) -> str:
    """Letters only: drop tone marks and circumflex/breve/horn, keep đ distinct from d."""
    s = unicodedata.normalize("NFD", s.lower())
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def _core(tok: str) -> str:
    return "".join(VI_WORD.findall(tok)).lower()


def _with_case(src: str, word: str) -> str:
    """Put `word` in place of the letters of token `src`, keeping its capitalisation and punctuation."""
    m = VI_WORD.search(src)
    if not m:
        return src
    w = word.capitalize() if m.group(0)[0].isupper() else word
    return src[:m.start()] + w + src[m.end():]


def reconcile_row(khn: KHN, nd: NomDict, key: str, img: np.ndarray, ink: np.ndarray, box: List[List[int]],
                  nom: str, qn: str, qn_readings: List[str], vertical: bool = False) -> Tuple[str, str, List[dict]]:
    """Resolve positions where the Nôm character and the Quốc ngữ word disagree (dịch âm, 1 char = 1 word).

    Every replacement must come from the images:
      1. a Nôm character read by another KimHanNom run (line / 0.6x / binarized / single cell) whose
         reading is the OCR'd word;
      2. a word read by another VietOCR variant that is a reading of the Nôm character;
      3. tone/diacritic restoration only: the OCR'd letters are kept and the marks are taken from the
         unique reading of the Nôm character with exactly those letters (script fonts lose marks).
    Anything else stays as read (and is coloured red).
    """
    chars, toks = list(nom), qn.split()
    if len(chars) != len(toks) or not chars:
        return nom, qn, []
    bad = [i for i, (c, t) in enumerate(zip(chars, toks)) if not nd.match(c, _core(t))]
    if not bad:
        return nom, qn, []
    p = np.array(box)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    line = img[y0:y1, x0:x1]
    order = (lambda k: k[1][:, 1].min()) if vertical else (lambda k: k[1][:, 0].min())
    nom_runs = []
    for tag, crop in (("line", white_pad(line, 150)),
                      ("line0.6", cv2.resize(white_pad(line, 150), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)),
                      ("linebin", cv2.copyMakeBorder(binarize(line), 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=(255, 255, 255)))):
        nom_runs.append(nom_line("".join(t for t, _, _ in sorted(khn.ocr(f"{key}|{tag}", crop, vertical), key=order))))
    cells = char_cells(ink[y0:y1, x0:x1], vertical)
    qn_runs = [r.split() for r in qn_readings if len(r.split()) == len(toks)]
    log = []
    for i in bad:
        alts = {r[i] for r in nom_runs if len(r) == len(chars)}
        if len(cells) == len(chars):
            a, b = cells[i]
            cell = line[:, a:b] if not vertical else line[a:b, :]
            alts |= set(nom_line("".join(t for t, _, _ in khn.ocr(f"{key}|cell{i}", white_pad(cell, 60), False)))[:1])
        alts.discard(chars[i])
        w = _core(toks[i])
        hit = sorted(c for c in alts if nd.match(c, w))
        if hit:
            log.append({"pos": i, "side": "nom", "from": chars[i], "to": hit[0], "rule": "khn_alt"})
            chars[i] = hit[0]
            continue
        words = sorted({_core(r[i]) for r in qn_runs} - {w})
        hit_w = [v for v in words if nd.match(chars[i], v)]
        if hit_w:
            log.append({"pos": i, "side": "qn", "from": toks[i], "to": hit_w[0], "rule": "vietocr_alt"})
            toks[i] = _with_case(toks[i], hit_w[0])
            continue
        # Only when the OCR variants disagree on the marks: a unanimous reading is trusted over the dictionary
        unanimous = len({_core(r[i]) for r in qn_runs}) <= 1
        same_letters = {key_to_text(k) for k in nd.c2k.get(chars[i], ()) if strip_marks(key_to_text(k)) == strip_marks(w)}
        if len(same_letters) == 1 and not unanimous:
            new = same_letters.pop()
            log.append({"pos": i, "side": "qn", "from": toks[i], "to": new, "rule": "diacritic_restore"})
            toks[i] = _with_case(toks[i], new)
    return "".join(chars), " ".join(toks), log


# ---------------------------------------------------------------------------
# Book 1: Tế cô hồn văn (table: verse number | Nôm | Quốc ngữ)
# ---------------------------------------------------------------------------
def build_te_co_hon(khn: KHN, voter: QNVoter, work: Path):
    title = TE_CO_HON
    pdf = OUTPUT_DIR / title / f"{title}.pdf"
    pages = render_pdf(pdf, work / title, title)
    folders: Dict[str, List[str]] = {"ch": [], "vi": [], "misc": []}
    ch_labels, vi_labels, records, audit = {}, {}, [], []

    for pno, name, path in pages:
        img = cv2_imread_unicode(path)
        h, w = img.shape[:2]
        ink = ink_mask(img)
        bands = runs(ink.sum(0) > h * 0.002, min_len=40, max_gap=60)
        if len(bands) < 2 or bands[0][1] > w * 0.2:
            folders["misc"].append(name)
            continue
        num_b = bands[0]
        # Only rows carrying a printed verse number are verses; restricting the column profile to
        # them keeps the centred closing title/credits of the last page from bridging the columns.
        num_rows = runs(ink[:, num_b[0]:num_b[1]].sum(1) > 3, min_len=15, max_gap=12)
        row_mask = np.zeros(h, bool)
        for a, b in num_rows:
            row_mask[max(0, a - 20):b + 20] = True
        bands = runs(ink[row_mask].sum(0) > row_mask.sum() * 0.002, min_len=40, max_gap=60)
        if len(bands) < 3:
            folders["misc"].append(name)
            continue
        nom_b, qn_b = bands[1], bands[-1]
        rows = runs(ink[:, qn_b[0]:qn_b[1]].sum(1) > 3, min_len=25, max_gap=12)
        rows = [(a, b) for a, b in rows
                if ink[a:b, num_b[0]:num_b[1]].sum() > 30 and ink[a:b, nom_b[0]:nom_b[1]].sum() > 200]
        if not rows:
            folders["misc"].append(name)
            continue
        x0, x1 = max(0, nom_b[0] - 15), min(w, nom_b[1] + 15)
        khn_items = khn.ocr(f"{title}|{name}|nomcol|h", img[:, x0:x1], vertical=False)
        ch_boxes, vi_boxes = [], []
        for idx, (y0, y1) in enumerate(rows, start=1):
            yc = (y0 + y1) / 2
            hits = [k for k in khn_items if k[1][:, 1].min() - 10 <= yc <= k[1][:, 1].max() + 10]
            hits.sort(key=lambda k: k[1][:, 0].min())
            cb = tight_box(ink, nom_b[0], y0, nom_b[1], y1)
            nom, nom_info = read_nom_line(khn, f"{title}|{name}|row{idx}", img, ink, cb,
                                          nom_line("".join(k[0] for k in hits)))
            qcrop = img[max(0, y0 - 8):min(h, y1 + 8), max(0, qn_b[0] - 10):min(w, qn_b[1] + 10)]
            readings = voter.read_all(qcrop)
            qn_raw, info = voter.vote(readings, HAN_RE.findall(nom))
            nom, qn_raw, fixes = reconcile_row(khn, voter.nd, f"{title}|{name}|row{idx}", img, ink, cb, nom,
                                               qn_raw, readings)
            qn = TextCleaner.clean_quocngu(qn_raw)
            vb = tight_box(ink, qn_b[0], y0, qn_b[1], y1)
            ch_boxes.append({"transcription": nom, "points": cb})
            vi_boxes.append({"transcription": qn, "points": vb})
            records.append({"img_id": name, "sentence_id": f"{Path(name).stem}.{idx}",
                            "SinoNom_Char": nom, "ChuQN_txt": qn})
            audit.append({"img_id": name, "row": idx, "nom": nom, "khn_conf": [round(k[2], 3) for k in hits],
                          "qn": qn, "qn_variants": readings, "fixes": fixes, **nom_info, **info})
        folders["ch"].append(name)
        folders["vi"].append(name)
        ch_labels[name], vi_labels[name] = ch_boxes, vi_boxes
        print(f"  p{pno:03d}: {len(rows)} verses")

    write_audit(title, audit)
    return write_book(title, pages, folders, ch_labels, vi_labels, records, pdf)


# ---------------------------------------------------------------------------
# Book 2: Tho Ho Xuan Huong [chu nom] (each Nôm line is followed by its Quốc ngữ line)
# ---------------------------------------------------------------------------
QN_LINE_PROB = 0.8  # VietOCR confidence: Quốc ngữ lines >= 0.88, Nôm lines <= 0.67 on this book


def build_tho_hxh(khn: KHN, voter: QNVoter, work: Path):
    title = THO_HXH
    pdf = OUTPUT_DIR / title / f"{title}.pdf"
    pages = render_pdf(pdf, work / title, title)
    stream = []  # reading order across pages, so a verse split by a page break is still paired
    page_img = {}
    for pno, name, path in pages:
        if pno == 1:  # illustrated cover
            continue
        img = cv2_imread_unicode(path)
        ink = ink_mask(img, 150)
        page_img[name] = (img, ink)
        for y0, y1 in runs(ink.sum(1) > 2, min_len=30, max_gap=6):
            xs = np.flatnonzero(ink[y0:y1].sum(0))
            box = tight_box(ink, int(xs.min()), y0, int(xs.max()) + 1, y1)
            p = np.array(box)
            crop = img[p[:, 1].min():p[:, 1].max(), p[:, 0].min():p[:, 0].max()]
            text, prob = voter.models[0].predict(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), return_prob=True)
            stream.append({"img": name, "box": box, "kind": "qn" if prob >= QN_LINE_PROB else "nom",
                           "prob": float(prob), "text0": text})

    ch_labels: Dict[str, list] = {}
    vi_labels: Dict[str, list] = {}
    records, audit, dropped = [], [], []
    sent_no: Counter = Counter()
    i = 0
    while i < len(stream):
        cur = stream[i]
        nxt = stream[i + 1] if i + 1 < len(stream) else None
        if cur["kind"] != "nom" or nxt is None or nxt["kind"] != "qn":
            dropped.append({"img": cur["img"], "kind": cur["kind"], "text0": cur["text0"], "why": "unpaired"})
            i += 1
            continue
        letters = "".join(VI_WORD.findall(nxt["text0"]))
        if letters and letters.isupper() and len(letters) > 1:  # poem / book title pair
            dropped.append({"img": cur["img"], "kind": "title", "text0": nxt["text0"], "why": "title"})
            i += 2
            continue
        img, ink = page_img[cur["img"]]
        name = cur["img"]
        key = f"{title}|{name}|y{cur['box'][0][1]}"
        p = np.array(cur["box"])
        y_c = (p[:, 1].min() + p[:, 1].max()) / 2
        # one horizontal OCR per page; its boxes are matched to the Nôm line by vertical overlap
        page_items = khn.ocr(f"{title}|{name}|page|h", img, False)
        hits = [k for k in page_items if k[1][:, 1].min() - 10 <= y_c <= k[1][:, 1].max() + 10]
        first = nom_line("".join(t for t, _, _ in sorted(hits, key=lambda k: k[1][:, 0].min())))
        nom, nom_info = read_nom_line(khn, key, img, ink, cur["box"], first)
        q_img, _ = page_img[nxt["img"]]
        qp = np.array(nxt["box"])
        qcrop = q_img[max(0, qp[:, 1].min() - 6):qp[:, 1].max() + 6, max(0, qp[:, 0].min() - 8):qp[:, 0].max() + 8]
        readings = voter.read_all(qcrop)
        qn_raw, info = voter.vote(readings, HAN_RE.findall(nom))
        nom, qn_raw, fixes = reconcile_row(khn, voter.nd, key, img, ink, cur["box"], nom, qn_raw, readings)
        qn = TextCleaner.clean_quocngu(qn_raw)
        if not nom or not qn:
            dropped.append({"img": name, "kind": "pair", "text0": nxt["text0"], "why": "empty OCR"})
            i += 2
            continue
        sent_no[name] += 1
        ch_labels.setdefault(name, []).append({"transcription": nom, "points": cur["box"]})
        vi_labels.setdefault(nxt["img"], []).append({"transcription": qn, "points": nxt["box"]})
        records.append({"img_id": name, "sentence_id": f"{Path(name).stem}.{sent_no[name]}",
                        "SinoNom_Char": nom, "ChuQN_txt": qn})
        audit.append({"img_id": name, "row": sent_no[name], "qn_img": nxt["img"], "nom": nom, "qn": qn,
                      "qn_variants": readings, "fixes": fixes, **nom_info, **info})
        i += 2

    names = [n for _, n, _ in pages]
    folders = {"ch": [n for n in names if n in ch_labels], "vi": [n for n in names if n in vi_labels]}
    folders["misc"] = [n for n in names if n not in ch_labels and n not in vi_labels]
    write_audit(title, audit + [{"dropped": d} for d in dropped])
    return write_book(title, pages, folders, ch_labels, vi_labels, records, pdf)


# ---------------------------------------------------------------------------
# Books 3+4: ThuDạLữHoàiNgâm (woodblock, Tnom) and its Hán-Việt reading Thụ Dạ…_Dịch (Tqn)
# ---------------------------------------------------------------------------
def _split_binding(img: np.ndarray) -> int:
    """x of the book binding: the darkest full-height strip near the middle of a 2-page scan."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray[int(h * 0.2):int(h * 0.8)] < 60).sum(0).astype(float)
    lo, hi = int(w * 0.35), int(w * 0.65)
    return lo + int(np.argmax(np.convolve(dark[lo:hi], np.ones(15), "same")))


def _register_gap(ink: np.ndarray) -> Optional[int]:
    """y between the upper (Hán) and lower (Nôm) registers of one page, or None if single block."""
    h, w = ink.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1)))
    prof = ((ink > 0) & (rules == 0)).sum(1).astype(float)
    smooth = np.convolve(prof, np.ones(41) / 41, "same")  # scans are speckled: compare to the text rows
    lo, hi = int(h * 0.40), int(h * 0.62)
    gy = lo + int(np.argmin(smooth[lo:hi]))
    body = smooth[int(h * 0.2):int(h * 0.8)]
    return gy if smooth[gy] < 0.25 * np.median(body) else None


def _ocr_columns(khn: KHN, key: str, img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                 lang_type: int) -> List[dict]:
    """Vertical KimHanNom OCR of one region; returns columns ordered right-to-left in image coords."""
    region = img[y0:y1, x0:x1]
    cols = []
    for t, pts, conf in khn.ocr(key, region, vertical=True, lang_type=lang_type):
        text = nom_line(t)
        if not text:
            continue
        pts = pts + np.array([x0, y0])
        q = [[int(round(px)), int(round(py))] for px, py in pts]
        cols.append({"text": text, "points": q, "conf": conf, "x": float(pts[:, 0].mean())})
    return sorted(cols, key=lambda c: -c["x"])


def _align_chars_to_verses(chars: List[str], verses: List[List[str]], nd: NomDict):
    """Needleman-Wunsch between the OCR'd character stream and the verse syllables.

    Returns, for every verse, the [start, end) character span it covers (None if nothing aligned).
    """
    syl, owner = [], []
    for vi, v in enumerate(verses):
        syl += v
        owner += [vi] * len(v)
    n, m = len(chars), len(syl)
    GAP, MATCH, SUB = -1.0, 2.0, -0.6
    score = np.zeros((n + 1, m + 1), np.float32)
    back = np.zeros((n + 1, m + 1), np.int8)
    score[1:, 0] = GAP * np.arange(1, n + 1)
    score[0, 1:] = GAP * np.arange(1, m + 1)
    back[1:, 0], back[0, 1:] = 1, 2
    repeat = [j > 0 and owner[j] == owner[j - 1] and syl[j] == syl[j - 1] for j in range(m)]
    match = np.array([[MATCH if (nd.match(c, s) or (c in ITERATION_MARKS and repeat[j])) else SUB
                       for j, s in enumerate(syl)] for c in chars], np.float32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (score[i - 1, j - 1] + match[i - 1, j - 1], score[i - 1, j] + GAP, score[i, j - 1] + GAP)
            k = int(np.argmax(opts))
            score[i, j], back[i, j] = opts[k], k
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        k = back[i, j]
        if k == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    spans: List[Optional[List[int]]] = [None] * len(verses)
    for ci, sj in pairs:
        v = owner[sj]
        if spans[v] is None:
            spans[v] = [ci, ci + 1]
        else:
            spans[v][0] = min(spans[v][0], ci)
            spans[v][1] = max(spans[v][1], ci + 1)
    return spans, [(ci, syl[sj], repeat[sj]) for ci, sj in pairs]


def _fix_woodblock_chars(khn: KHN, nd: NomDict, stream: list, pairs: list, page_path: Dict[str, Path]) -> List[dict]:
    """Second opinions for aligned characters that do not read as their Hán-Việt syllable.

    Alternatives come only from other KimHanNom readings of the same image: the column re-read
    binarized / at 0.6x / in Nôm mode, and the single glyph cell. An alternative is taken only if it
    reads as the syllable; otherwise the first reading stays (and is coloured red).
    """
    import difflib
    imgs: Dict[str, np.ndarray] = {}
    col_alts: Dict[str, List[str]] = {}
    fixes = []
    for ci, syl, is_repeat in pairs:
        col, k = stream[ci]
        cur = col["text"][k]
        if nd.match(cur, syl) or (is_repeat and cur in ITERATION_MARKS):
            continue
        if col["img"] not in imgs:
            imgs[col["img"]] = cv2_imread_unicode(page_path[col["img"]])
        p = np.array(col["points"])
        x0, y0 = np.maximum(p.min(0), 0)
        x1, y1 = p.max(0)
        crop = imgs[col["img"]][y0:y1, x0:x1]
        if col["key"] not in col_alts:
            texts = []
            for tag, im, lt in (
                    ("bin", cv2.copyMakeBorder(binarize(crop), 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=(255, 255, 255)), 1),
                    ("s0.6", cv2.resize(white_pad(crop, 80), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), 1),
                    ("nom", white_pad(crop, 80), 2)):
                items = khn.ocr(f"{col['key']}|{tag}", im, vertical=True, lang_type=lt)
                texts.append(nom_line("".join(t for t, _, _ in sorted(items, key=lambda q: q[1][:, 1].min()))))
            col_alts[col["key"]] = texts
        alts = set()
        for t in col_alts[col["key"]]:
            if len(t) == len(col["text"]):
                alts.add(t[k])
                continue
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, col["text"], t, autojunk=False).get_opcodes():
                if op in ("equal", "replace") and i2 - i1 == j2 - j1 and i1 <= k < i2:
                    alts.add(t[j1 + k - i1])
        hit = sorted(a for a in alts if a != cur and (nd.match(a, syl) or (is_repeat and a in ITERATION_MARKS)))
        if not hit:
            cells = char_cells(ink_mask(crop, 150), vertical=True)
            if len(cells) == len(col["text"]):
                a, b = cells[k]
                one = khn.ocr(f"{col['key']}|cell{k}", white_pad(crop[a:b, :], 60), False, lang_type=1)
                c1 = nom_line("".join(t for t, _, _ in one))[:1]
                if c1 and c1 != cur and nd.match(c1, syl):
                    hit = [c1]
        if hit:
            fixes.append({"img": col["img"], "col": col["text"], "pos": k, "from": cur, "to": hit[0], "syl": syl})
            col["text"] = col["text"][:k] + hit[0] + col["text"][k + 1:]
    return fixes


def build_thu_da(khn: KHN, voter: QNVoter, work: Path):
    nd = voter.nd
    # --- Quốc ngữ: text layer of the _Dịch PDF (line boxes scaled to the rendered images) ---
    d_title = THU_DA_DICH
    d_pdf = OUTPUT_DIR / d_title / f"{d_title}.pdf"
    d_pages = render_pdf(d_pdf, work / d_title, d_title)
    doc = pymupdf.open(d_pdf)
    qn_lines = []  # (img name, text, points)
    d_vi_labels: Dict[str, list] = {}
    for (pno, name, path), pg in zip(d_pages, doc):
        scale = cv2_imread_unicode(path).shape[1] / pg.rect.width
        for blk in pg.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                text = TextCleaner.clean_quocngu("".join(s["text"] for s in ln["spans"]))
                if not VI_WORD.search(text):
                    continue
                x0, y0, x1, y1 = (int(round(v * scale)) for v in ln["bbox"])
                pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                qn_lines.append((name, text, pts))
                d_vi_labels.setdefault(name, []).append({"transcription": text, "points": pts})
    title_line, verses_qn = qn_lines[0], qn_lines[1:]  # first line is the work's title

    # --- Hán/Nôm: KimHanNom on every woodblock register ---
    title = THU_DA
    pdf = OUTPUT_DIR / title / f"{title}.pdf"
    pages = render_pdf(pdf, work / title, title)
    ch_labels: Dict[str, list] = {}
    stream: List[Tuple[str, str]] = []  # (char, image name) of the upper (Hán) register in reading order
    content = [(pno, name, path) for pno, name, path in pages if pno not in (1, 2, 10)]  # covers, title page
    layout = {}
    for pno, name, path in content:
        img = cv2_imread_unicode(path)
        bx = _split_binding(img)
        halves, gaps = {}, {}
        for side, (x0, x1) in (("right", (bx + 20, img.shape[1])), ("left", (0, bx - 20))):
            ink = ink_mask(img[:, x0:x1], 150)
            if ink.mean() >= 0.01:  # skip the black inside-cover half
                halves[side] = (x0, x1)
                gaps[side] = _register_gap(ink)
        layout[name] = (halves, gaps)
    # Every spread is printed from the same woodblock frame: an undetected register gap (show-through
    # on the last page) falls back to the facing page, then to the median over the book.
    found = [g for _, gaps in layout.values() for g in gaps.values() if g is not None]
    book_gap = int(np.median(found)) if found else None
    for pno, name, path in content:
        img = cv2_imread_unicode(path)
        h = img.shape[0]
        halves, gaps = layout[name]
        for side in halves:
            if gaps[side] is None:
                gaps[side] = next((g for g in gaps.values() if g is not None), book_gap)
        page_boxes = {"left": [], "right": []}
        upper_by_side = {}
        for side, (x0, x1) in halves.items():
            gy = gaps[side]
            regions = [("upper", 0, gy), ("lower", gy, h)] if gy else [("upper", 0, h)]
            for reg, y0, y1 in regions:
                # upper register: Hán text (read in Hán-Việt by the _Dịch book); lower register: Nôm
                cols = _ocr_columns(khn, f"{title}|{name}|{side}|{reg}", img, x0, y0, x1, y1,
                                    lang_type=1 if reg == "upper" else 2)
                for c in cols:
                    c["img"], c["key"] = name, f"{title}|{name}|{side}|{reg}|x{int(c['x'])}"
                page_boxes[side] += cols
                if reg == "upper":
                    upper_by_side[side] = cols
        # Content order of a traditional book: right page, then left page
        for side in ("right", "left"):
            for c in upper_by_side.get(side, []):
                stream += [(c, k) for k in range(len(c["text"]))]
        # Box order required by the rules for 2-page scans: left page first, then the right page
        if page_boxes["left"] or page_boxes["right"]:
            ch_labels[name] = page_boxes["left"] + page_boxes["right"]
    verses = [[w.lower() for w in VI_WORD.findall(t)] for _, t, _ in verses_qn]
    chars = [c["text"][k] for c, k in stream]
    spans, pairs = _align_chars_to_verses(chars, verses, nd)
    fixes = _fix_woodblock_chars(khn, nd, stream, pairs, {p[1]: p[2] for p in pages})
    chars = [c["text"][k] for c, k in stream]

    records, d_records, audit = [], [], []
    sent_no: Counter = Counter()
    d_sent_no: Counter = Counter()
    for vi, ((d_img, qn, _), span) in enumerate(zip(verses_qn, spans)):
        if span is None:
            audit.append({"verse": vi + 1, "qn": qn, "why": "no aligned characters"})
            continue
        nom = "".join(chars[span[0]:span[1]])
        img_name = stream[span[0]][0]["img"]
        sent_no[img_name] += 1
        d_sent_no[d_img] += 1
        records.append({"img_id": img_name, "sentence_id": f"{Path(img_name).stem}.{sent_no[img_name]}",
                        "SinoNom_Char": nom, "ChuQN_txt": qn})
        d_records.append({"img_id": d_img, "sentence_id": f"{Path(d_img).stem}.{d_sent_no[d_img]}",
                          "SinoNom_Char": nom, "ChuQN_txt": qn})
        audit.append({"verse": vi + 1, "img_id": img_name, "nom": nom, "qn": qn})
    unaligned = len(chars) - sum(s[1] - s[0] for s in spans if s)
    audit.append({"stream_chars": len(chars), "chars_outside_verses": unaligned, "title_line_dropped": title_line[1],
                  "char_fixes": fixes})
    ch_labels = {n: [{"transcription": c["text"], "points": c["points"]} for c in cols] for n, cols in ch_labels.items()}

    names = [n for _, n, _ in pages]
    folders = {"ch": [n for n in names if n in ch_labels], "vi": []}
    folders["misc"] = [n for n in names if n not in ch_labels]
    write_audit(title, audit)
    ev = write_book(title, pages, folders, ch_labels, {}, records, pdf)
    d_names = [n for _, n, _ in d_pages]
    write_book(d_title, d_pages, {"ch": [], "vi": [n for n in d_names if n in d_vi_labels],
                                   "misc": [n for n in d_names if n not in d_vi_labels]},
               {}, d_vi_labels, d_records, d_pdf)
    return ev


def write_audit(title: str, rows: List[dict]):
    out = ROOT / "reports" / "printed_rebuild" / f"{title}.audit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


BUILDERS = {"te_co_hon": build_te_co_hon, "tho_hxh": build_tho_hxh, "thu_da": build_thu_da}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="all", choices=["all", *BUILDERS])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offline", action="store_true", help="only use cached KimHanNom results")
    args = ap.parse_args()
    nd = NomDict.load()
    khn = KHN()
    khn.offline = args.offline
    voter = QNVoter(nd, device=args.device)
    work = Path(tempfile.gettempdir()) / "printed_rebuild_pages"
    for key, fn in BUILDERS.items():
        if args.book in ("all", key):
            print(f"\n=== {key} ===")
            fn(khn, voter, work)


if __name__ == "__main__":
    main()
