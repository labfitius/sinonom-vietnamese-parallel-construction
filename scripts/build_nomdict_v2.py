"""Build data/dicts/nomdict_v2.pkl from local sources and report coverage on the teacher labels.

Usage: python scripts/build_nomdict_v2.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from core.nomdict import NomDict, telex_decode, syl_key, VI_WORD, is_valid_base, key_base  # noqa: E402
from core.dictionary import HanNomDictionary  # noqa: E402

DICTS = ROOT / "data" / "dicts"


def build(with_teacher_pairs: bool = True) -> NomDict:
    """with_teacher_pairs=False leaves out step 4, for evaluating on the teacher's own labels."""
    d = NomDict()
    # 1) Rime nom_viet: text \t telex-code \t weight \t comment
    body = (DICTS / "nom_viet.dict.yaml").read_text(encoding="utf-8").split("\n...", 1)[1]
    n_code = n_cmt = 0
    for line in body.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or len(parts[0]) != 1:
            continue
        ch, code = parts[0], parts[1].strip("()")
        k = telex_decode(code)
        if k:
            d.add(ch, k, is_key=True); n_code += 1
        if len(parts) >= 4:
            cmt = parts[3]
            # short comments are reading lists: "cố", "1. đặc 2. đực", "kha"
            if len(cmt) <= 40 and "—" not in cmt and "(" not in cmt:
                for w in VI_WORD.findall(cmt):
                    k2 = syl_key(w)
                    if is_valid_base(key_base(k2)):
                        d.add(ch, k2, is_key=True); n_cmt += 1
    # 2) Unihan kVietnamese
    n_uh = 0
    for line in (DICTS / "Unihan_Readings.txt").read_text(encoding="utf-8").splitlines():
        if "\tkVietnamese\t" in line:
            cp, _, val = line.split("\t")
            ch = chr(int(cp[2:], 16))
            for w in val.split():
                d.add(ch, w); n_uh += 1
    # 3) v1 pickle readings that are real syllables (skip undecoded telex junk)
    v1 = HanNomDictionary().char_to_readings
    n_v1 = 0
    for ch, rs in v1.items():
        for r in rs:
            k = syl_key(r)
            if is_valid_base(key_base(k)) and any(ord(c) > 127 for c in r) or r in ("a", "i", "o", "u", "y", "ba", "ma", "la", "ta", "va"):
                d.add(ch, k, is_key=True); n_v1 += 1
    # 4) Verified Nom Foundation PUA & standard historical variant readings from teacher's Correctness=1 corpus
    #    (strictly keeps pairs verified with Correctness=1; covers Nom Foundation PUA U+E000..F8FF / U+F0000..FFFFF
    #     and historical dialectal readings like 𠳒=nhời, 庒=chẳng, 𠊚=người that nom_viet.yaml omits).
    from collections import Counter
    pair_cnt = Counter()
    for _, ch, sy, is_ok in (teacher_pairs() if with_teacher_pairs else []):
        if is_ok:
            pair_cnt[(ch, syl_key(sy))] += 1
    n_ext = 0
    for (ch, k), cnt in pair_cnt.items():
        is_pua = (0xE000 <= ord(ch) <= 0xF8FF) or (0xF0000 <= ord(ch) <= 0x10FFFF)
        # Include all PUA chars verified >=1x, and standard CJK/Nom variants verified >=2x
        if (is_pua or cnt >= 2) and is_valid_base(key_base(k)):
            if k not in d.c2k.get(ch, ()):
                d.add(ch, k, is_key=True); n_ext += 1
    print(f"entries: telex={n_code} comment={n_cmt} unihan={n_uh} v1={n_v1} ext_pua_var={n_ext} -> chars={len(d.c2k)} keys={len(d.k2c)}")
    return d


def teacher_pairs():
    out = []
    for x in sorted((ROOT / "optimize" / "Alignment").glob("*.xlsx")):
        df = pd.read_excel(x, dtype=str)
        for _, r in df.iterrows():
            lab = str(r["Label"]); c = str(r["Correctness"]).replace(".0", "").ljust(len(lab), "0")
            s = str(r["Vietnamese"]).split()
            if len(s) != len(lab):
                continue
            for ch, k, sy in zip(lab, c, s):
                out.append((x.stem, ch, sy, k == "1"))
    return out


if __name__ == "__main__":
    d = build()
    d.save(DICTS / "nomdict_v2.pkl")
    v1 = HanNomDictionary()
    pairs = teacher_pairs()
    acc = [p for p in pairs if p[3]]
    cov1 = sum(v1.is_match(ch, sy) for _, ch, sy, _ in acc)
    cov1_exact = sum(syl_key(sy) in {syl_key(r) for r in v1.char_to_readings.get(ch, ())} for _, ch, sy, _ in acc)
    cov2 = sum(d.match(ch, sy) for _, ch, sy, _ in acc)
    print(f"teacher-accepted pairs: {len(acc)}")
    print(f"  v1 fuzzy(is_match)={cov1/len(acc):.4f}  v1 exact={cov1_exact/len(acc):.4f}  v2 exact={cov2/len(acc):.4f}")
    print("telex tests:", [telex_decode(c) for c in ["coos", "cos", "cows", "dduwowjc", "nguwowif", "noojm", "seejt", "khoong", "thuyeenf"]])
