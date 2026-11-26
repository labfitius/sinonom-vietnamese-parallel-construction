"""
Nom dictionary v2: tone-aware, tone-placement-agnostic Han-Nom <-> Quoc ngu readings.

Sources (all local, see scripts/build_nomdict_v2.py):
  - data/dicts/nom_viet.dict.yaml : Rime dictionary, readings Telex-encoded in the `code` column
                                    (e.g. 固 coos/cos/cows -> cố/có/cớ). The v1 pickle stored many of
                                    these codes undecoded ("noojm"), which is why it missed readings
                                    such as 固=có, 特=được, 𠊚=người.
  - data/dicts/Unihan_Readings.txt: kVietnamese field.
  - optional extra pairs (char, syllable) mined from verified labels.

Matching uses `syl_key`: lowercase NFC syllable with the tone mark removed and appended as a
digit, so "hoà" and "hòa" compare equal while "có" and "cố" stay different.
"""
from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

TONE_MARKS = {"\u0300": "2", "\u0301": "1", "\u0309": "3", "\u0303": "4", "\u0323": "5"}
TELEX_TONE = {"f": "2", "s": "1", "r": "3", "x": "4", "j": "5"}
TONE_TO_MARK = {v: k for k, v in TONE_MARKS.items()}

_ONSET = r"(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnprstvx])?"
_NUC = r"[aăâeêioôơuưy]{1,3}"
_CODA = r"(ch|ng|nh|[cmnpt])?"
SYL_RE = re.compile(f"^{_ONSET}{_NUC}{_CODA}$")
VI_WORD = re.compile(r"[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+")


def syl_key(s: str) -> str:
    """Canonical key: base letters (with ^ breve horn kept) + tone digit (0..5)."""
    s = unicodedata.normalize("NFD", s.strip().lower())
    tone = "0"
    out = []
    for ch in s:
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
        else:
            out.append(ch)
    base = unicodedata.normalize("NFC", "".join(out))
    return base + tone


def key_base(k: str) -> str:
    return k[:-1]


def is_valid_base(base: str) -> bool:
    return bool(SYL_RE.match(base))


def telex_decode(code: str) -> Optional[str]:
    """Decode a Telex code to a syllable key (base+tone digit). Returns None if not a valid syllable."""
    code = code.strip().lower()
    if not code or not code.isalpha():
        return None
    out: list = []
    tone = "0"
    vowels = set("aăâeêioôơuưy")
    for i, c in enumerate(code):
        has_vowel = any(x in vowels for x in out)
        if c in TELEX_TONE and has_vowel:
            tone = TELEX_TONE[c]
            continue
        if c == "z" and has_vowel:
            tone = "0"
            continue
        if c == "d" and out and out[-1] == "d" and len(out) == 1:
            out[-1] = "đ"; continue
        if c in "aeo" and out:
            # circumflex: aa, ee, oo (allow a consonant coda in between: "coon" handled by search back)
            j = len(out) - 1
            while j >= 0 and out[j] not in vowels:
                j -= 1
            if j >= 0 and out[j] == c and (j == len(out) - 1 or i == len(code) - 1 or True):
                if j == len(out) - 1:
                    out[j] = {"a": "â", "e": "ê", "o": "ô"}[c]; continue
        if c == "w":
            j = len(out) - 1
            while j >= 0 and out[j] not in vowels:
                j -= 1
            if j >= 0:
                m = {"a": "ă", "o": "ơ", "u": "ư", "ô": "ơ", "â": "ă"}
                if out[j] in m:
                    out[j] = m[out[j]]
                    if out[j] == "ơ" and j >= 1 and out[j - 1] == "u":
                        out[j - 1] = "ư"
                    continue
                if out[j] == "ơ" and j >= 1 and out[j - 1] == "u":
                    out[j - 1] = "ư"; continue
            out.append("ư"); continue
        out.append(c)
    base = "".join(out)
    if base.startswith("uw"):
        return None
    if not is_valid_base(base):
        return None
    return base + tone


def key_to_text(k: str) -> str:
    """Render a key back to a syllable with the tone mark on the conventional vowel."""
    base, tone = k[:-1], k[-1]
    if tone == "0":
        return base
    nfd = unicodedata.normalize("NFD", base)
    letters = list(base)
    vidx = [i for i, ch in enumerate(letters) if ch in "aăâeêioôơuưy"]
    # skip glide u in qu-, i in gi-
    if base.startswith("qu") and len(vidx) > 1 and vidx[0] == 1:
        vidx = vidx[1:]
    if base.startswith("gi") and len(vidx) > 1 and vidx[0] == 1:
        vidx = vidx[1:]
    if not vidx:
        return base
    pos = None
    for i in vidx:
        if letters[i] in "ăâêôơư":
            pos = i
    if pos is None:
        has_coda = vidx[-1] < len(letters) - 1
        pos = vidx[-1] if (len(vidx) == 1 or has_coda or len(vidx) == 3) else vidx[-2]
        if len(vidx) == 3:
            pos = vidx[1]
    ch = unicodedata.normalize("NFD", letters[pos]) + TONE_TO_MARK[tone]
    letters[pos] = unicodedata.normalize("NFC", ch)
    return "".join(letters)


class NomDict:
    """char -> set of syllable keys, key -> set of chars."""

    def __init__(self, c2k: Optional[Dict[str, Set[str]]] = None):
        self.c2k: Dict[str, Set[str]] = c2k or {}
        self.k2c: Dict[str, Set[str]] = {}
        self._reindex()

    def _reindex(self):
        self.k2c = {}
        for c, ks in self.c2k.items():
            for k in ks:
                self.k2c.setdefault(k, set()).add(c)

    def add(self, ch: str, syl_or_key: str, is_key: bool = False):
        k = syl_or_key if is_key else syl_key(syl_or_key)
        if not is_valid_base(key_base(k)):
            return
        self.c2k.setdefault(ch, set()).add(k)
        self.k2c.setdefault(k, set()).add(ch)

    def has(self, ch: str) -> bool:
        return ch in self.c2k

    def match(self, ch: str, syl: str) -> bool:
        return syl_key(syl) in self.c2k.get(ch, ())

    def chars_for(self, syl: str) -> Set[str]:
        return self.k2c.get(syl_key(syl), set())

    def merged(self, extra: Iterable[tuple]) -> "NomDict":
        c2k = {c: set(v) for c, v in self.c2k.items()}
        d = NomDict(c2k)
        for ch, syl in extra:
            d.add(ch, syl)
        return d

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"c2k": self.c2k}, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NomDict":
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "dicts" / "nomdict_v2.pkl"
        with open(path, "rb") as f:
            return cls(pickle.load(f)["c2k"])
