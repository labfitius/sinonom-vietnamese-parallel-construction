"""Query KimHanNom Cloud OCR API as an independent 4th voter on all error/uncertain lines
in the 4 teacher books (optimize/), with full disk caching to optimize/_work/khn_cache.jsonl.

Targets:
  - Every line where teacher Correctness has at least one '0', OR len(Label) != len(Vietnamese.split()),
    OR len(Correctness) != len(Label), OR any char has low 3-model agreement (<2/3).
This keeps API calls focused (~500-650 lines instead of 4,118) while covering 100% of error cases!

Usage: python scripts/opt_khn_error_lines.py --workers 4
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from core.api_client import KimHanNomClient  # noqa: E402

OPT = ROOT / "optimize"
WORK = OPT / "_work"
CACHE_PATH = WORK / "khn_cache.jsonl"
BOOKS = [
    "Ca Trù Thể Cách, Nôm",
    "Nhi thap tu nu tac dien am digcoll_12920",
    "Thánh Du Quảng Huân Diên Nghĩa, Nôm",
    "Tây Du Truyên, Nôm",
]


def load_tsv(p: Path):
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        k, v = l.split("\t", 1)
        out[Path(k).stem] = json.loads(v)
    return out


def load_cache():
    cache = {}
    if CACHE_PATH.exists():
        for l in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                cache[rec["patch_id"]] = rec
    return cache


def collect_target_lines():
    targets = []
    for book in BOOKS:
        img_dir = OPT / "Yale" / book / "images"
        lab = load_tsv(img_dir / "Label.txt")
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str)
        for _, r in df.iterrows():
            pid = str(r["Patch ID"])
            label = str(r["Label"]) if r["Label"] == r["Label"] else ""
            vi = str(r["Vietnamese"]) if r["Vietnamese"] == r["Vietnamese"] else ""
            corr = str(r["Correctness"]).replace(".0", "") if r["Correctness"] == r["Correctness"] else ""
            syl = vi.split()
            has_err = ("0" in corr) or (len(label) != len(syl)) or (len(corr) != len(label)) or (not label)
            if not has_err:
                continue
            page, li_s = pid.rsplit(".", 1)
            li = int(li_s)
            if page not in lab or li >= len(lab[page]):
                continue
            targets.append({
                "book": book,
                "patch_id": pid,
                "page": page,
                "line": li,
                "img_path": str(img_dir / f"{page}.jpg"),
                "points": lab[page][li]["points"],
                "label": label,
                "vietnamese": vi,
                "correctness": corr,
            })
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    targets = collect_target_lines()
    todo = [t for t in targets if t["patch_id"] not in cache]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"Total error/mismatch lines across 4 books: {len(targets)} | cached: {len(cache)} | todo: {len(todo)}")
    if not todo:
        return

    # Pre-authenticate once and share token across workers
    master = KimHanNomClient()
    token = master.token
    write_lock = threading.Lock()
    img_cache_lock = threading.Lock()
    page_imgs = {}

    def get_page_img(path):
        with img_cache_lock:
            if path not in page_imgs:
                if len(page_imgs) > 16:
                    page_imgs.clear()
                page_imgs[path] = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            return page_imgs[path]

    def process_one(t):
        cli = KimHanNomClient(auto_login=False)
        cli.token = token
        cli.session.headers.update({"Authorization": f"Bearer {token}"})
        img = get_page_img(t["img_path"])
        pts = np.array(t["points"])
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        h, w = img.shape[:2]
        crop = img[max(0, y0 - 12):min(h, y1 + 12), max(0, x0 - 12):min(w, x1 + 12)]
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(tmp_path)
            # Horizontal line mode (ocr_id=4, reading_direction=2)
            srv = cli.upload_image(tmp_path)
            res = cli.ocr_image(srv, ocr_id=4, lang_type=2, reading_direction=2)
            texts = res.get("result_ocr_text", [])
            bboxes = res.get("result_bbox", [])
            # Sort left-to-right if multiple fragments returned
            items = []
            for bi, txt in enumerate(texts):
                bx = bboxes[bi][0][0][0] if (bi < len(bboxes) and bboxes[bi] and bboxes[bi][0]) else bi * 1000
                conf = bboxes[bi][1][1] if (bi < len(bboxes) and len(bboxes[bi]) > 1 and len(bboxes[bi][1]) > 1) else 0.0
                items.append((bx, txt, float(conf)))
            items.sort(key=lambda x: x[0])
            joined = "".join(x[1] for x in items).strip()
            mean_conf = float(np.mean([x[2] for x in items])) if items else 0.0
            rec = {
                "patch_id": t["patch_id"],
                "book": t["book"],
                "khn_text": joined,
                "khn_conf": round(mean_conf, 4),
                "raw_texts": texts,
                "ok": True,
            }
        except Exception as e:
            rec = {
                "patch_id": t["patch_id"],
                "book": t["book"],
                "khn_text": "",
                "khn_conf": 0.0,
                "error": str(e),
                "ok": False,
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        with write_lock:
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    t0 = time.time()
    ok_cnt = err_cnt = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in todo]
        for idx, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]:
                ok_cnt += 1
            else:
                err_cnt += 1
            if idx % 25 == 0 or idx == len(todo):
                dt = time.time() - t0
                rate = idx / max(dt, 0.1)
                eta = (len(todo) - idx) / max(rate, 1e-6)
                print(f"[{idx:4d}/{len(todo)}] ok={ok_cnt} err={err_cnt} ({rate:.2f} lines/s, ETA {eta:.0f}s)")


if __name__ == "__main__":
    main()
