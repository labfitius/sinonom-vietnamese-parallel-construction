"""Ensemble + Sound-Constrained Decoding + KimHanNom 4th-Voter + QN Typo Detection
for the 4 teacher books in optimize/.

Combines (all zero page-leakage):
  1. Teacher 3 models: char_det.txt (vote), ConvNeXt, ConvNeXtFinetune
  2. Our 5-fold Out-of-Fold (OOF) ConvNeXt-Tiny classifier (full-dictionary sound-constrained logits)
  3. KimHanNom Cloud OCR API (4th independent voter on all error/mismatch lines)
  4. NomDict v2 (Telex-decoded nom_viet + Unihan kVietnamese) + cross-fold PUA readings

Outputs:
  - optimize/improved/<book>.xlsx (teacher schema + audit columns)
  - optimize/improved/ned_summary.xlsx (before/after comparison per book)
  - optimize/improved/gold_review_300.xlsx (300 stratified lines for human gold check)

Usage: python scripts/opt_ensemble_improve.py
"""
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from core.nomdict import NomDict, syl_key, key_base  # noqa: E402

OPT = ROOT / "optimize"
WORK = OPT / "_work"
OUT_IMP = OPT / "improved"
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


def load_oof():
    """Returns dict[(book, page, line, pos)] -> oof record."""
    oof = {}
    for p in sorted(WORK.glob("oof_fold*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            oof[(r["book"], r["page"], int(r["line"]), int(r["pos"]))] = r
    return oof


def load_khn():
    khn = {}
    p = WORK / "khn_cache.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("ok") and r.get("khn_text"):
                    khn[r["patch_id"]] = r
    return khn


def align_khn_to_label(label: str, khn_text: str):
    """Map each position i in label to the aligned KimHanNom character (or None)."""
    aligned = [None] * len(label)
    if not khn_text or not label:
        return aligned
    if len(khn_text) == len(label):
        return list(khn_text)
    sm = difflib.SequenceMatcher(None, label, khn_text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                aligned[i1 + k] = khn_text[j1 + k]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                aligned[i1 + k] = khn_text[j1 + k]
    return aligned


def edit_dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_cross_page_readings():
    """For zero-leakage PUA/variant lookup: map (char, syl_key) -> set of pages where teacher accepted it."""
    pair_pages = defaultdict(set)
    for book in BOOKS:
        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str)
        for _, r in df.iterrows():
            pid = str(r["Patch ID"])
            page = pid.rsplit(".", 1)[0]
            lab = str(r["Label"]) if r["Label"] == r["Label"] else ""
            corr = str(r["Correctness"]).replace(".0", "").ljust(len(lab), "0")
            syl = str(r["Vietnamese"]).split() if r["Vietnamese"] == r["Vietnamese"] else []
            if len(syl) != len(lab):
                continue
            for ch, c, s in zip(lab, corr, syl):
                if c == "1":
                    pair_pages[(ch, syl_key(s))].add(page)
    return pair_pages


def main():
    OUT_IMP.mkdir(parents=True, exist_ok=True)
    nd = NomDict.load()
    oof = load_oof()
    khn = load_khn()
    pair_pages = build_cross_page_readings()
    print(f"Loaded OOF crops: {len(oof)} | KimHanNom cached error lines: {len(khn)} | NomDict chars: {len(nd.c2k)}")

    def is_exact_dict(ch: str, syl: str, page: str) -> bool:
        sk = syl_key(syl)
        if nd.match(ch, syl):
            return True
        # Zero-leakage cross-page verified PUA/reading (must appear on >=2 OTHER pages)
        other_pages = pair_pages.get((ch, sk), set()) - {page}
        return len(other_pages) >= 2

    def is_fuzzy_dict(ch: str, syl: str) -> bool:
        kb = key_base(syl_key(syl))
        for k in nd.c2k.get(ch, ()):
            if key_base(k) == kb:
                return True
        return False

    summary_rows = []
    gold_pool = []

    for book in BOOKS:
        # Load teacher crops npz for aligned teacher 3-model candidates
        d = np.load(WORK / f"crops_{book}.npz", allow_pickle=True)
        crop_cands = {}
        for i in range(len(d["label"])):
            key = (str(d["page"][i]), int(d["line"][i]), int(d["pos"][i]))
            crop_cands[key] = json.loads(str(d["cands"][i]))

        df = pd.read_excel(OPT / "Alignment" / f"{book}.xlsx", dtype=str)
        out_rows = []
        stats = Counter()

        for _, r in df.iterrows():
            pid = str(r["Patch ID"])
            page, li_s = pid.rsplit(".", 1)
            li = int(li_s)
            vi = str(r["Vietnamese"]) if r["Vietnamese"] == r["Vietnamese"] else ""
            lab_old = str(r["Label"]) if r["Label"] == r["Label"] else ""
            corr_raw = str(r["Correctness"]).replace(".0", "") if r["Correctness"] == r["Correctness"] else ""
            corr_old = corr_raw.rjust(len(lab_old), "0") if len(corr_raw) < len(lab_old) else corr_raw[:len(lab_old)]
            syls = vi.split()

            khn_rec = khn.get(pid)
            khn_text = khn_rec["khn_text"] if khn_rec else ""
            khn_conf = khn_rec["khn_conf"] if khn_rec else 0.0

            new_chars = list(lab_old)
            new_corr = list(corr_old)
            sources = []
            typo_flags = []

            # Case 1: length mismatch repair using KimHanNom full-line OCR
            if len(lab_old) != len(syls) and khn_text and len(khn_text) == len(syls):
                old_matches = sum(is_exact_dict(c, s, page) for c, s in zip(lab_old, syls))
                khn_matches = sum(is_exact_dict(c, s, page) for c, s in zip(khn_text, syls))
                if khn_matches >= old_matches + 1 and khn_matches >= max(1, int(0.75 * len(syls))):
                    new_chars = list(khn_text)
                    new_corr = ["1" if is_exact_dict(c, s, page) else "0" for c, s in zip(khn_text, syls)]
                    sources.append(f"len_repair_khn({len(lab_old)}->{len(khn_text)})")
                    stats["len_repaired_lines"] += 1

            # Case 2: aligned per-character optimization
            if len(new_chars) == len(syls):
                khn_aligned = align_khn_to_label("".join(new_chars), khn_text)
                for pos in range(len(new_chars)):
                    stats["total_chars"] += 1
                    ch_old = new_chars[pos]
                    c_old = new_corr[pos] if pos < len(new_corr) else "0"
                    syl = syls[pos].lower()

                    if c_old == "1":
                        stats["teacher_ok"] += 1
                        stats["new_ok"] += 1
                        continue

                    stats["teacher_err"] += 1
                    oof_rec = oof.get((book, page, li, pos))
                    t_cands = crop_cands.get((page, li, pos))
                    c_khn = khn_aligned[pos] if pos < len(khn_aligned) else None

                    # Build candidate score maps
                    t_maps = []
                    if t_cands:
                        for labs, scs in t_cands:
                            t_maps.append(dict(zip(labs, scs)))
                    else:
                        t_maps = [{}, {}, {}]

                    o_top = dict(zip(oof_rec["top_labels"], oof_rec["top_scores"])) if oof_rec else {}
                    o_ex = dict(oof_rec["dict_exact"]) if oof_rec else {}
                    o_fz = dict(oof_rec["dict_fuzzy"]) if oof_rec else {}

                    # Check QN Typo Suspect: strong 4-model consensus on ch_old + close reading
                    t_top1s = [list(m.keys())[0] if m else None for m in t_maps]
                    o_top1 = oof_rec["top_labels"][0] if (oof_rec and oof_rec["top_labels"]) else None
                    o_p1 = oof_rec["top_scores"][0] if (oof_rec and oof_rec["top_scores"]) else 0.0
                    t0_p1 = t_maps[0].get(ch_old, 0.0)
                    if (ch_old == o_top1 and all(x == ch_old for x in t_top1s if x)
                            and o_p1 >= 0.72 and t0_p1 >= 0.65):
                        # Check if ch_old has a reading within edit distance 1 of syl
                        close_rds = [k for k in nd.c2k.get(ch_old, ()) if edit_dist(key_base(k), key_base(syl_key(syl))) <= 1]
                        if close_rds:
                            typo_flags.append(f"pos{pos}:{ch_old}({syl})")
                            stats["qn_typo_flagged"] += 1
                            continue

                    # Collect ALL exact-reading candidates across OOF full-dict + teacher top-10 + KimHanNom
                    all_exact = set(o_ex.keys())
                    for m in t_maps:
                        for c in m:
                            if is_exact_dict(c, syl, page):
                                all_exact.add(c)
                    if c_khn and is_exact_dict(c_khn, syl, page):
                        all_exact.add(c_khn)

                    best_c, best_score, best_src = None, 0.0, ""
                    for c in all_exact:
                        p_o = max(o_ex.get(c, 0.0), o_top.get(c, 0.0))
                        p_t0 = t_maps[0].get(c, 0.0)
                        p_t1 = t_maps[1].get(c, 0.0)
                        p_t2 = t_maps[2].get(c, 0.0)
                        p_k = khn_conf if (c_khn == c) else 0.0
                        # Weighted visual fusion
                        sc = 0.40 * p_o + 0.20 * p_t0 + 0.12 * p_t1 + 0.12 * p_t2 + 0.25 * p_k
                        # Determine strongest evidence source
                        if c_khn == c and (p_o >= 0.02 or p_t0 > 0 or p_k >= 0.55):
                            src = "khn+oof" if p_o >= 0.02 else "khn_4th_voter"
                            sc += 0.15  # cross-architecture corroboration bonus
                        elif p_o >= 0.05:
                            src = "oof_sound_decode"
                        elif max(p_t0, p_t1, p_t2) >= 0.05:
                            src = "teacher_top10_dictv2"
                        else:
                            src = "weak_dict"
                        if sc > best_score:
                            best_c, best_score, best_src = c, sc, src

                    # Accept exact-dict fix if visual support exceeds principled absolute OR relative-to-top1 threshold
                    accepted = False
                    if best_c is not None:
                        p_o = max(o_ex.get(best_c, 0.0), o_top.get(best_c, 0.0))
                        p_t_max = max(t_maps[0].get(best_c, 0.0), t_maps[1].get(best_c, 0.0), t_maps[2].get(best_c, 0.0))
                        rel_o = p_o / max(o_p1, 1e-6) if o_p1 > 0 else 0.0
                        in_o_top20 = best_c in (oof_rec["top_labels"][:20] if oof_rec else [])
                        if (
                            (best_c == c_khn and khn_conf >= 0.50)
                            or p_o >= 0.06
                            or (in_o_top20 and rel_o >= 0.18 and p_o >= 0.01)
                            or (p_o >= 0.012 and p_t_max >= 0.01)
                            or best_score >= 0.07
                        ):
                            new_chars[pos] = best_c
                            new_corr[pos] = "1"
                            sources.append(f"p{pos}:{ch_old}->{best_c}[{best_src}:{best_score:.2f}]")
                            stats[f"fixed_{best_src}"] += 1
                            stats["fixed_total"] += 1
                            stats["new_ok"] += 1
                            accepted = True

                    # Fallback: tone-fuzzy candidate with strong independent visual agreement (e.g., hoi/nga dialect)
                    if not accepted:
                        all_fz = set(o_fz.keys())
                        if c_khn and is_fuzzy_dict(c_khn, syl):
                            all_fz.add(c_khn)
                        best_fc, best_fs = None, 0.0
                        for c in all_fz:
                            p_o = max(o_fz.get(c, 0.0), o_top.get(c, 0.0))
                            p_k = khn_conf if (c_khn == c) else 0.0
                            fs = 0.55 * p_o + 0.45 * p_k
                            if fs > best_fs:
                                best_fc, best_fs = c, fs
                        if best_fc is not None and (best_fs >= 0.35 or (best_fc == c_khn and o_fz.get(best_fc, 0.0) >= 0.10)):
                            new_chars[pos] = best_fc
                            new_corr[pos] = "1"
                            sources.append(f"p{pos}:{ch_old}->{best_fc}[fuzzy:{best_fs:.2f}]")
                            stats["fixed_fuzzy"] += 1
                            stats["fixed_total"] += 1
                            stats["new_ok"] += 1

            lab_new = "".join(new_chars)
            corr_new = "".join(new_corr)
            changed = (lab_new != lab_old)
            if changed:
                stats["changed_lines"] += 1

            row_out = {
                "ID": r["ID"],
                "Patch ID": pid,
                "Vietnamese": vi,
                "Label": lab_new,
                "Correctness": corr_new,
                "Label_Teacher": lab_old,
                "Correctness_Teacher": corr_old,
                "Changed": "YES" if changed else "NO",
                "Change_Source": "; ".join(sources),
                "QN_Typo_Flag": "; ".join(typo_flags),
                "KimHanNom_Raw": khn_text,
            }
            out_rows.append(row_out)
            gold_pool.append((book, row_out))

        out_df = pd.DataFrame(out_rows)
        # Save improved xlsx with string Correctness preserved (no float loss!)
        out_xlsx = OUT_IMP / f"{book}.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Improved Alignment"
        cols = list(out_df.columns)
        ws.append(cols)
        hdr_font = Font(name="Arial", size=11, bold=True)
        hdr_fill = PatternFill("solid", fgColor="FFE6EEF8")
        for ci in range(1, len(cols) + 1):
            c = ws.cell(1, ci); c.font = hdr_font; c.fill = hdr_fill
        chg_fill = PatternFill("solid", fgColor="FFE8F5E9")
        for ri, row in enumerate(out_rows, start=2):
            for ci, col in enumerate(cols, start=1):
                cell = ws.cell(ri, ci, value=str(row[col]))
                cell.number_format = "@"  # force text so leading 0s in Correctness are never stripped!
                if row["Changed"] == "YES" and col in ("Label", "Correctness", "Changed", "Change_Source"):
                    cell.fill = chg_fill
        wb.save(str(out_xlsx))

        # Compute full-book metrics (including length-mismatch rows)
        t_ok = sum(r["Correctness_Teacher"].count("1") for r in out_rows)
        t_tot = sum(len(r["Label_Teacher"]) for r in out_rows)
        n_ok = sum(r["Correctness"].count("1") for r in out_rows)
        n_tot = sum(len(r["Label"]) for r in out_rows)
        t_acc = 100.0 * t_ok / max(t_tot, 1)
        n_acc = 100.0 * n_ok / max(n_tot, 1)
        err_red = 100.0 * ((t_tot - t_ok) - (n_tot - n_ok)) / max(t_tot - t_ok, 1)

        summary_rows.append({
            "Book": book,
            "Lines": len(out_rows),
            "Total_Chars": n_tot,
            "Teacher_OK": t_ok,
            "Teacher_Err": t_tot - t_ok,
            "Teacher_Acc_%": round(t_acc, 3),
            "Improved_OK": n_ok,
            "Improved_Err": n_tot - n_ok,
            "Improved_Acc_%": round(n_acc, 3),
            "Delta_Acc_%": round(n_acc - t_acc, 3),
            "Error_Reduction_%": round(err_red, 2),
            "Fixed_OOF_Sound": stats["fixed_oof_sound_decode"],
            "Fixed_KHN_4th": stats["fixed_khn_4th_voter"] + stats["fixed_khn+oof"],
            "Fixed_Top10_DictV2": stats["fixed_teacher_top10_dictv2"],
            "Fixed_Fuzzy": stats["fixed_fuzzy"],
            "Len_Repaired_Lines": stats["len_repaired_lines"],
            "QN_Typo_Flagged": stats["qn_typo_flagged"],
        })
        print(f"\n[{book}]")
        print(f"  Teacher:  {t_ok}/{t_tot} = {t_acc:.3f}% (err={t_tot - t_ok})")
        print(f"  Improved: {n_ok}/{n_tot} = {n_acc:.3f}% (err={n_tot - n_ok}) | +{n_acc - t_acc:.3f}% (error reduction {err_red:.1f}%)")
        print(f"  Breakdown: oof_sound={stats['fixed_oof_sound_decode']} khn={stats['fixed_khn_4th_voter']+stats['fixed_khn+oof']} "
              f"top10_v2={stats['fixed_teacher_top10_dictv2']} fuzzy={stats['fixed_fuzzy']} len_rep={stats['len_repaired_lines']} qn_typo={stats['qn_typo_flagged']}")

    sum_df = pd.DataFrame(summary_rows)
    # Append OVERALL row
    tot_t_ok = sum_df["Teacher_OK"].sum(); tot_t_err = sum_df["Teacher_Err"].sum()
    tot_n_ok = sum_df["Improved_OK"].sum(); tot_n_err = sum_df["Improved_Err"].sum()
    tot_c = sum_df["Total_Chars"].sum()
    ov = {
        "Book": "OVERALL (4 BOOKS)",
        "Lines": int(sum_df["Lines"].sum()),
        "Total_Chars": int(tot_c),
        "Teacher_OK": int(tot_t_ok),
        "Teacher_Err": int(tot_t_err),
        "Teacher_Acc_%": round(100.0 * tot_t_ok / (tot_t_ok + tot_t_err), 3),
        "Improved_OK": int(tot_n_ok),
        "Improved_Err": int(tot_n_err),
        "Improved_Acc_%": round(100.0 * tot_n_ok / tot_c, 3),
        "Delta_Acc_%": round(100.0 * tot_n_ok / tot_c - 100.0 * tot_t_ok / (tot_t_ok + tot_t_err), 3),
        "Error_Reduction_%": round(100.0 * (tot_t_err - tot_n_err) / max(tot_t_err, 1), 2),
        "Fixed_OOF_Sound": int(sum_df["Fixed_OOF_Sound"].sum()),
        "Fixed_KHN_4th": int(sum_df["Fixed_KHN_4th"].sum()),
        "Fixed_Top10_DictV2": int(sum_df["Fixed_Top10_DictV2"].sum()),
        "Fixed_Fuzzy": int(sum_df["Fixed_Fuzzy"].sum()),
        "Len_Repaired_Lines": int(sum_df["Len_Repaired_Lines"].sum()),
        "QN_Typo_Flagged": int(sum_df["QN_Typo_Flagged"].sum()),
    }
    sum_df = pd.concat([sum_df, pd.DataFrame([ov])], ignore_index=True)
    sum_df.to_excel(OUT_IMP / "ned_summary.xlsx", index=False)
    sum_df.to_csv(OUT_IMP / "ned_summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== OVERALL SUMMARY =====")
    print(sum_df.to_string(index=False))

    # Build 300-line stratified human gold review workbook (75 per book: up to 50 changed + 25 unchanged)
    rng = np.random.default_rng(42)
    gold_rows = []
    for book in BOOKS:
        b_rows = [r for bk, r in gold_pool if bk == book]
        chg = [r for r in b_rows if r["Changed"] == "YES"]
        unchg = [r for r in b_rows if r["Changed"] == "NO"]
        pick_chg = list(rng.choice(len(chg), size=min(50, len(chg)), replace=False)) if chg else []
        n_rem = 75 - len(pick_chg)
        pick_un = list(rng.choice(len(unchg), size=min(n_rem, len(unchg)), replace=False)) if unchg else []
        for idx in pick_chg:
            r = dict(chg[idx]); r["Book"] = book; r["Human_Gold_Label"] = ""; r["Verdict_Improved_Better"] = ""
            gold_rows.append(r)
        for idx in pick_un:
            r = dict(unchg[idx]); r["Book"] = book; r["Human_Gold_Label"] = ""; r["Verdict_Improved_Better"] = ""
            gold_rows.append(r)
    pd.DataFrame(gold_rows).to_excel(OUT_IMP / "gold_review_300.xlsx", index=False)
    print(f"Wrote {OUT_IMP / 'gold_review_300.xlsx'} ({len(gold_rows)} lines)")


if __name__ == "__main__":
    main()
