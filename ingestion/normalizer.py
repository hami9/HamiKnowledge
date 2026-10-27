"""Text normalization for English, Persian, and mixed-language technical text.

Preserves raw verbatim text for citations while producing a clean, normalized
form for lexical search indexing and retrieval matching.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode character points
ZWNJ = "\u200c"
ZWSP = "\u200b"
ARABIC_TATWEEL = "\u0640"

# Arabic diacritics / Harakat
_DIACRITICS_RE = re.compile(r"[\u064b-\u0652\u0656-\u065f\u0670]")

# Character translation table for Persian normalization
_CHAR_REPLACEMENTS: dict[str, str] = {
    "\u064a": "\u06cc",  # Arabic Yeh (ي) -> Persian Yeh (ی)
    "\u0649": "\u06cc",  # Alef Maksura (ى) -> Persian Yeh (ی)
    "\u0643": "\u06a9",  # Arabic Kaf (ك) -> Persian Keheh (ک)
    "\u0629": "\u0647",  # Teh Marbuta (ة) -> Heh (ه)
    "\u06c0": f"\u0647{ZWNJ}\u06cc",  # Heh with Yeh above (ۀ) -> ه + ZWNJ + ی
    # Arabic/Eastern digits to standard digits for search uniformity
    "\u0660": "0", "\u06f0": "0",
    "\u0661": "1", "\u06f1": "1",
    "\u0662": "2", "\u06f2": "2",
    "\u0663": "3", "\u06f3": "3",
    "\u0664": "4", "\u06f4": "4",
    "\u0665": "5", "\u06f5": "5",
    "\u0666": "6", "\u06f6": "6",
    "\u0667": "7", "\u06f7": "7",
    "\u0668": "8", "\u06f8": "8",
    "\u0669": "9", "\u06f9": "9",
}

# Regex to clean up spaces and zero-width characters
_MULTIPLE_ZWNJ_RE = re.compile(f"{ZWNJ}+")
_ZWNJ_AROUND_SPACE_RE = re.compile(rf"\s+{ZWNJ}|\s*{ZWNJ}\s+")
_MULTIPLE_SPACES_RE = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


class PersianNormalizer:
    """Standardizes Persian and Arabic text variants for search indexing.

    Raw text is never mutated for citation storage; this normalizer is used to
    produce the search vector / query string.
    """

    def clean_persian_letters(self, text: str) -> str:
        """Replace Arabic letter variants with Persian counterparts and standardize ZWNJ."""
        if not text:
            return ""

        # Normalize Unicode representation to NFC
        text = unicodedata.normalize("NFC", text)

        # Remove tatweel and zero-width spaces
        text = text.replace(ARABIC_TATWEEL, "").replace(ZWSP, "")

        # Apply character replacements (ي -> ی, ك -> ک, ة -> ه)
        for src, dst in _CHAR_REPLACEMENTS.items():
            if src in text:
                text = text.replace(src, dst)

        # Remove diacritics (Harakat)
        text = _DIACRITICS_RE.sub("", text)

        # Collapse multiple ZWNJs and clean ZWNJ around whitespace
        text = _MULTIPLE_ZWNJ_RE.sub(ZWNJ, text)
        text = _ZWNJ_AROUND_SPACE_RE.sub(" ", text)

        return text

    def normalize_for_search(self, text: str) -> str:
        """Full search normalization: letter cleaning, whitespace collapse, and NFC."""
        cleaned = self.clean_persian_letters(text)
        # Strip trailing/leading spaces from lines
        cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
        cleaned = _MULTIPLE_NEWLINES_RE.sub("\n\n", cleaned)
        cleaned = _MULTIPLE_SPACES_RE.sub(" ", cleaned)
        return cleaned.strip()


def normalize_search_query(query: str) -> str:
    """Helper to prepare a query string for lexical or hybrid search."""
    return PersianNormalizer().normalize_for_search(query)
