"""Improve the teacher's handwritten-OCR labels in optimize/ (v2: stricter, separately reported).

Voter groups (independent recognisers of the character image):
  T : the teacher's 3 models (char_det.txt, .ConvNeXt, .ConvNeXtFinetune), top-1 of any of them
  O : our 5-fold out-of-fold ConvNeXt-Tiny (page-split, never saw the page), top-1
  K : KimHanNom, line OCR aligned to the position or single-character OCR of the crop

Only characters the teacher marked 0 are touched. Three kinds of change are counted separately:
  1. label_fix   : the Nôm label is replaced by a character c != old such that
                   - c is top-1 in at least 2 of the 3 groups T, O, K, and
                   - c reads as the Quốc ngữ syllable in a dictionary built WITHOUT the teacher's labels
                     (nom_viet + Unihan + v1), or the pair is accepted by the teacher on >= 2 other pages.
  2. qn_typo     : the old label is confirmed by all 3 groups with high confidence and only the tone of the
                   syllable differs -> Correctness=1 per the teacher's README convention (Quốc ngữ typo).
  3. dict_cover  : label unchanged and it reads as the syllable in that dictionary (the teacher's dictionary
                   lacked the reading).
Lines whose label length differs from the syllable count are replaced by the KimHanNom line OCR only when
it has the right length and clearly more dictionary matches (len_repair).

Outputs optimize/improved_v2/<book>.xlsx, summary.xlsx, review_changes.png sheets.
Usage: python scripts/opt_improve_v2.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from core.nomdict import key_base, syl_key  # noqa: E402
from build_nomdict_v2 import build  # noqa: E402
from opt_ensemble_improve import BOOKS, align_khn_to_label, load_khn, load_oof  # noqa: E402

OPT = ROOT / "optimize"
WORK = OPT / "_work"
OUT = OPT / "improved_v2"
HIGH_T, HIGH_O, HIGH_K = 0.65, 0.72, 0.5


def load_khn_char():
    out = {}
    p = WORK / "khn_char_cache.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("ok") and r.get("text"):
                    out[r["key"]] = r
    return out


def cross_page_pairs():
    """(char, syl_key) -> pages on which the teacher accepted the pair (Correctness=1)."""
    pages = defaultdict(set)
    for book in BOOKS:
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str).fillna("")
        for _, r in df.iterrows():
            lab, syl = r["Label"], r["Vietnamese"].split()
            corr = r["Correctness"].replace(".0", "").ljust(len(lab), "0")
            if len(syl) != len(lab):
                continue
            for ch, c, s in zip(lab, corr, syl):
                if c == "1":
                    pages[(ch, syl_key(s))].add(r["Patch ID"].rsplit(".", 1)[0])
    return pages


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nd = build(with_teacher_pairs=False)
    pair_pages = cross_page_pairs()
    oof, khn_line, khn_char = load_oof(), load_khn(), load_khn_char()
    print(f"dict chars={len(nd.c2k)} | OOF={len(oof)} | KHN lines={len(khn_line)} | KHN chars={len(khn_char)}")

    def reads(ch, syl, page):
        """Dictionary check that never uses the teacher's label of this page."""
        return nd.match(ch, syl) or len(pair_pages.get((ch, syl_key(syl)), set()) - {page}) >= 2

    summary, review = [], []
    for book in BOOKS:
        z = np.load(WORK / f"crops_{book}.npz", allow_pickle=True)
        cands = {(str(p), int(l), int(q)): json.loads(str(c)) for p, l, q, c in zip(z["page"], z["line"], z["pos"], z["cands"])}
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str).fillna("")
        rows, st = [], Counter()
        for _, r in df.iterrows():
            pid = r["Patch ID"]
            page, li = pid.rsplit(".", 1)
            li = int(li)
            syls = r["Vietnamese"].split()
            old = r["Label"]
            corr_raw = r["Correctness"].replace(".0", "")
            corr_old = corr_raw.rjust(len(old), "0") if len(corr_raw) < len(old) else corr_raw[:len(old)]
            lab, corr, notes = list(old), list(corr_old), []
            st["chars"] += len(old)
            st["teacher_ok"] += corr_old.count("1")
            kl = khn_line.get(pid)

            if len(old) != len(syls):
                kt = kl["khn_text"] if kl else ""
                if kt and len(kt) == len(syls) and kl["khn_conf"] >= 0.6:
                    old_m = sum(reads(c, s, page) for c, s in zip(old, syls))
                    new_m = sum(reads(c, s, page) for c, s in zip(kt, syls))
                    if new_m >= max(old_m + 1, corr_old.count("1") + 1, int(0.75 * len(syls))):
                        lab = list(kt)
                        corr = ["1" if reads(c, s, page) else "0" for c, s in zip(kt, syls)]
                        notes.append(f"len_repair:{old}->{kt}")
                        st["len_repair_lines"] += 1
            else:
                k_aligned = align_khn_to_label(old, kl["khn_text"]) if kl else [None] * len(old)
                for pos, (o, s) in enumerate(zip(old, syls)):
                    if corr_old[pos] == "1":
                        continue
                    s = s.lower()
                    tc = cands.get((page, li, pos)) or [[[], []]] * 3
                    t_top = {m[0][0]: m[1][0] for m in tc if m[0]}
                    orec = oof.get((book, page, li, pos))
                    o_top, o_p = (orec["top_labels"][0], orec["top_scores"][0]) if orec else (None, 0.0)
                    kc = khn_char.get(f"{page}|{li}|{pos}")
                    k_set = {x for x in (k_aligned[pos], kc["text"][:1] if kc and len(kc["text"]) == 1 else None) if x}
                    groups = [set(t_top), {o_top} - {None}, k_set]
                    support = Counter()
                    for g in groups:
                        for c in g:
                            support[c] += 1
                    fix = [c for c, n in support.most_common() if c != o and n >= 2 and reads(c, s, page)]
                    if fix:
                        c = fix[0]
                        lab[pos], corr[pos] = c, "1"
                        notes.append(f"p{pos}:{o}->{c}[label_fix T={sorted(t_top)} O={o_top} K={sorted(k_set)}]")
                        st["label_fix"] += 1
                        review.append((book, pid, pos, o, c, s))
                        continue
                    strong_old = (all(top == o and p >= HIGH_T for top, p in t_top.items()) and o_top == o
                                  and o_p >= HIGH_O and o in k_set and (kc is None or kc["conf"] >= HIGH_K))
                    same_letters = any(key_base(k) == key_base(syl_key(s)) for k in nd.c2k.get(o, ()))
                    if strong_old and same_letters:
                        corr[pos] = "1"
                        notes.append(f"p{pos}:{o}[qn_typo {s}]")
                        st["qn_typo"] += 1
                    elif reads(o, s, page):
                        corr[pos] = "1"
                        notes.append(f"p{pos}:{o}[dict_cover {s}]")
                        st["dict_cover"] += 1
            rows.append({"ID": r["ID"], "Patch ID": pid, "Vietnamese": r["Vietnamese"], "Label": "".join(lab),
                         "Correctness": "".join(corr), "Label_Teacher": old, "Correctness_Teacher": corr_old,
                         "Changed": "YES" if "".join(lab) != old else "NO", "Notes": "; ".join(notes)})
        pd.DataFrame(rows).to_excel(OUT / f"{book}.xlsx", index=False)
        n, ok = st["chars"], st["teacher_ok"]
        lf = st["label_fix"]
        len_gain = sum(r["Correctness"].count("1") - r["Correctness_Teacher"].count("1") for r in rows
                       if r["Notes"].startswith("len_repair"))
        summary.append({
            "Book": book, "Chars": n, "Teacher_Acc_%": round(100 * ok / n, 3),
            "Label_fixes": lf, "Len_repair_lines": st["len_repair_lines"],
            "Acc_after_label_fixes_%": round(100 * (ok + lf + len_gain) / n, 3),
            "QN_typo_convention": st["qn_typo"],
            "Acc_plus_qn_typo_%": round(100 * (ok + lf + len_gain + st["qn_typo"]) / n, 3),
            "Dict_coverage": st["dict_cover"],
            "Acc_plus_dict_%": round(100 * (ok + lf + len_gain + st["qn_typo"] + st["dict_cover"]) / n, 3),
        })
    sm = pd.DataFrame(summary)
    tot = sm.sum(numeric_only=True)
    n = tot["Chars"]
    ok = sum(r["Teacher_Acc_%"] * r["Chars"] / 100 for r in summary)
    sm.loc[len(sm)] = {"Book": "OVERALL", "Chars": n, "Teacher_Acc_%": round(100 * ok / n, 3),
                       "Label_fixes": tot["Label_fixes"], "Len_repair_lines": tot["Len_repair_lines"],
                       "Acc_after_label_fixes_%": round(sum(r["Acc_after_label_fixes_%"] * r["Chars"] for r in summary) / n, 3),
                       "QN_typo_convention": tot["QN_typo_convention"],
                       "Acc_plus_qn_typo_%": round(sum(r["Acc_plus_qn_typo_%"] * r["Chars"] for r in summary) / n, 3),
                       "Dict_coverage": tot["Dict_coverage"],
                       "Acc_plus_dict_%": round(sum(r["Acc_plus_dict_%"] * r["Chars"] for r in summary) / n, 3)}
    sm.to_excel(OUT / "summary.xlsx", index=False)
    json.dump(review, open(OUT / "label_fixes.json", "w", encoding="utf-8"), ensure_ascii=False)
    pd.set_option("display.width", 250)
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
