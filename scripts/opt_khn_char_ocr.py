"""KimHanNom single-character OCR of every character the teacher marked wrong (Correctness=0).

This adds a voter that is independent of the teacher's three models and of our ConvNeXt: the
full-resolution crop of the character (char_det.txt box) is sent alone to KimHanNom.

Output: optimize/_work/khn_char_cache.jsonl  {"key": "<page>|<line>|<pos>", "text": ..., "conf": ...}
Usage:  python -u scripts/opt_khn_char_ocr.py [--workers 2]
"""
import argparse
import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from core.api_client import KimHanNomClient  # noqa: E402
from opt_extract_crops import assign_lines, imread, load_tsv  # noqa: E402

OPT = ROOT / "optimize"
CACHE = OPT / "_work" / "khn_char_cache.jsonl"
MIN_INTERVAL = 2.5


def error_chars():
    """Yield (key, page image path, box points) for every Correctness=0 char of aligned rows."""
    for bdir in sorted((OPT / "Yale").iterdir()):
        book = bdir.name
        img_dir = bdir / "images"
        lab = load_tsv(img_dir / "Label.txt")
        det = load_tsv(img_dir / "char_det.txt")
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str)
        df[["page", "line"]] = df["Patch ID"].str.extract(r"^(.*)\.(\d{3})$")
        for page, g in df.groupby("page", sort=False):
            if page not in lab:
                continue
            per_line = assign_lines(lab[page], det.get(page, []))
            for _, r in g.iterrows():
                li = int(r["line"])
                label = str(r["Label"]) if r["Label"] == r["Label"] else ""
                corr = str(r["Correctness"]).replace(".0", "").ljust(len(label), "0")
                idx = per_line[li] if li < len(per_line) else []
                if not label or len(idx) != len(label):
                    continue
                for pos, ci in enumerate(idx):
                    if corr[pos] != "1":
                        yield f"{page}|{li}|{pos}", img_dir / f"{page}.jpg", np.array(det[page][ci]["points"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["key"] for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [t for t in error_chars() if t[0] not in done]
    print(f"error chars to OCR: {len(todo)} (cached {len(done)})")
    lock = threading.Lock()
    throttle = threading.Lock()
    last_call = [0.0]
    local = threading.local()
    pages = {}

    def run(item):
        key, img_path, pts = item
        while not hasattr(local, "cli"):
            try:
                local.cli = KimHanNomClient(auto_login=True)
            except Exception:
                time.sleep(10)
        with lock:
            if img_path not in pages:
                pages[img_path] = imread(img_path)
            gray = pages[img_path]
        x0, y0 = np.floor(pts.min(0)).astype(int) - 6
        x1, y1 = np.ceil(pts.max(0)).astype(int) + 6
        crop = gray[max(0, y0):y1, max(0, x0):x1]
        crop = cv2.copyMakeBorder(crop, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
        tmp = Path(tempfile.gettempdir()) / f"khn_char_{threading.get_ident()}.jpg"
        cv2.imencode(".jpg", crop)[1].tofile(str(tmp))
        res = None
        for attempt in range(6):
            with throttle:  # one call per MIN_INTERVAL across workers: bursts get the account refused
                time.sleep(max(0.0, last_call[0] + MIN_INTERVAL - time.time()))
                last_call[0] = time.time()
            try:
                res = local.cli.ocr_image(local.cli.upload_image(str(tmp)), ocr_id=4, lang_type=2, reading_direction=2)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
                try:
                    local.cli = KimHanNomClient(auto_login=True)
                except Exception:
                    pass
        texts = res["result_ocr_text"] if res else []
        confs = [b[1][1] for b in res["result_bbox"]] if res else []
        rec = {"key": key, "text": "".join(texts), "conf": max(confs) if confs else 0.0, "ok": res is not None}
        with lock:
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(run, todo), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
