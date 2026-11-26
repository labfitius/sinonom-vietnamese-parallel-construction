"""
Dictionary and Phonetic Verification Engine for Hán Nôm - Quốc ngữ
"""
import os
import re
import pickle
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
from core.config import HANNOM_DICT_PKL

class HanNomDictionary:
    _instance = None

    def __init__(self, pkl_path: Optional[str] = None):
        if pkl_path is None:
            pkl_path = str(HANNOM_DICT_PKL)
        self.pkl_path = pkl_path
        self.char_to_readings: Dict[str, Set[str]] = {}
        self.reading_to_chars: Dict[str, Set[str]] = {}
        self.compound_words: Dict[str, str] = {}
        self._load()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HanNomDictionary()
        return cls._instance

    def _load(self):
        # Prefer v2 decoded dictionary (nomdict_v2.pkl) alongside v1 fallback
        v2_path = os.path.join(os.path.dirname(self.pkl_path), "nomdict_v2.pkl")
        self._v2_c2k: Dict[str, Set[str]] = {}
        if os.path.exists(v2_path):
            with open(v2_path, 'rb') as f:
                self._v2_c2k = pickle.load(f).get('c2k', {})
        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.char_to_readings = data.get('char_to_readings', {})
                self.reading_to_chars = data.get('reading_to_chars', {})
                self.compound_words = data.get('compound_words', {})
        elif not self._v2_c2k:
            print(f"Warning: Dictionary file {self.pkl_path} not found.")
        # Merge v2 keys into char_to_readings for full backwards compatibility
        if self._v2_c2k:
            from core.nomdict import key_to_text
            for ch, ks in self._v2_c2k.items():
                s = self.char_to_readings.setdefault(ch, set())
                for k in ks:
                    txt = key_to_text(k)
                    s.add(txt)
                    self.reading_to_chars.setdefault(txt, set()).add(ch)

    @staticmethod
    def strip_accents(text: str) -> str:
        """Strip Vietnamese tone diacritics and normalize i/y for phonetic fuzzy matching."""
        s = unicodedata.normalize('NFD', text)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.replace('đ', 'd').replace('Đ', 'D')
        s = s.lower()
        # Normalize i/y at end of syllable (e.g. li <-> ly, ki <-> ky, mi <-> my)
        s = re.sub(r'([bcdfghklmnpqrstvx])y\b', r'\1i', s)
        return s

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize using NFC and lower."""
        return unicodedata.normalize('NFC', text.strip()).lower()

    def get_readings(self, char: str) -> List[str]:
        """Return list of readings for a single character."""
        char = unicodedata.normalize('NFC', char)
        return list(self.char_to_readings.get(char, []))

    def has_char(self, char: str) -> bool:
        """Check if character exists in dictionary."""
        char = unicodedata.normalize('NFC', char)
        return (char in self._v2_c2k) or (char in self.char_to_readings)

    def is_match(self, char: str, word: str) -> bool:
        """
        Check if a Hán Nôm character matches a Vietnamese word.
        Rule:
        1. Exact tone-aware match via nomdict_v2 syl_key (tone-placement agnostic: hòa == hoà).
        2. Exact match with any registered reading in v1 dictionary.
        3. Fuzzy stripped-accent match if base vowel/consonant matches.
        """
        char = unicodedata.normalize('NFC', char)
        word = self.normalize_text(word)
        if not char or not word:
            return False

        if self._v2_c2k and char in self._v2_c2k:
            from core.nomdict import syl_key
            if syl_key(word) in self._v2_c2k[char]:
                return True

        readings = self.char_to_readings.get(char, set())
        if not readings:
            return False

        if word in readings:
            return True

        # Check tone-stripped or fuzzy equivalence
        word_stripped = self.strip_accents(word)
        for r in readings:
            if word_stripped == self.strip_accents(r):
                return True

        return False

    def verify_sentence_pair(self, nom_str: str, qn_str: str) -> dict:
        """
        Verify alignment between a Hán Nôm sentence and a Quốc ngữ sentence.
        Returns match status per Nôm character AND per Quốc ngữ word for character-level XLSX coloring.
        """
        if not isinstance(nom_str, str):
            nom_str = "" if nom_str is None or (isinstance(nom_str, float) and str(nom_str) == "nan") else str(nom_str)
        if not isinstance(qn_str, str):
            qn_str = "" if qn_str is None or (isinstance(qn_str, float) and str(qn_str) == "nan") else str(qn_str)

        # Clean punctuation from Nôm: keep CJK, Extensions, and PUA (Nom Foundation) characters
        nom_chars = [
            c for c in nom_str
            if ('\u3400' <= c <= '\u9fff'
                or '\uf900' <= c <= '\ufaff'
                or '\ue000' <= c <= '\uf8ff'
                or '\U00020000' <= c <= '\U0003ffff'
                or '\U000f0000' <= c <= '\U0010ffff')
        ]

        # Tokenize Quốc ngữ into words
        qn_words = re.findall(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+', qn_str)
        qn_words = [self.normalize_text(w) for w in qn_words]

        nom_statuses = []  # (char, is_match, reason)
        word_statuses = [False] * len(qn_words)
        matched_count = 0
        total_count = len(nom_chars)

        if len(nom_chars) == len(qn_words):
            for idx, (c, w) in enumerate(zip(nom_chars, qn_words)):
                if not self.has_char(c):
                    nom_statuses.append((c, False, 'not_in_dict'))
                    word_statuses[idx] = False
                elif self.is_match(c, w):
                    nom_statuses.append((c, True, 'matched'))
                    word_statuses[idx] = True
                    matched_count += 1
                else:
                    nom_statuses.append((c, False, 'mismatch'))
                    word_statuses[idx] = False
        else:
            w_idx = 0
            for c in nom_chars:
                if not self.has_char(c):
                    nom_statuses.append((c, False, 'not_in_dict'))
                    continue
                matched = False
                for delta in range(max(0, w_idx - 1), min(len(qn_words), w_idx + 3)):
                    if self.is_match(c, qn_words[delta]):
                        matched = True
                        word_statuses[delta] = True
                        w_idx = delta + 1
                        break
                if matched:
                    nom_statuses.append((c, True, 'matched'))
                    matched_count += 1
                else:
                    nom_statuses.append((c, False, 'mismatch'))

        score = (matched_count / total_count * 100.0) if total_count > 0 else 0.0
        return {
            'total_chars': total_count,
            'matched_chars': matched_count,
            'mismatched_chars': total_count - matched_count,
            'similarity_pct': score,
            'is_valid': total_count > 0 and score >= 95.0,
            'char_statuses': nom_statuses,
            'word_statuses': word_statuses,
        }
