"""Tests for Persian, Arabic, and mixed-language text normalization."""

from __future__ import annotations

import unicodedata

from ingestion.normalizer import PersianNormalizer, normalize_search_query


def test_arabic_yeh_and_kaf_are_standardized_to_persian() -> None:
    normalizer = PersianNormalizer()
    arabic_input = "كتاب يوسف و على"
    expected = "کتاب یوسف و علی"
    assert normalizer.clean_persian_letters(arabic_input) == expected


def test_teh_marbuta_and_heh_with_yeh() -> None:
    normalizer = PersianNormalizer()
    arabic_input = "دائرة و برنامۀ"
    cleaned = normalizer.clean_persian_letters(arabic_input)
    assert "دائره" in cleaned
    assert "برنامه\u200cی" in cleaned


def test_zwnj_half_spaces_are_preserved_and_cleaned() -> None:
    normalizer = PersianNormalizer()
    text_with_multiple_zwnj = "می\u200c\u200cشود و می \u200cرود"
    cleaned = normalizer.clean_persian_letters(text_with_multiple_zwnj)
    assert "می\u200cشود" in cleaned
    assert "می رود" in cleaned


def test_mixed_latin_technical_identifiers_preserved_verbatim() -> None:
    normalizer = PersianNormalizer()
    mixed = "برای اجرای دستور kubectl get pods باید مقدار restartPolicy برابر Always باشد."
    cleaned = normalizer.clean_persian_letters(mixed)

    # Identifiers must remain unaltered in spelling and case
    assert "kubectl get pods" in cleaned
    assert "restartPolicy" in cleaned
    assert "Always" in cleaned


def test_full_search_normalization_collapses_whitespace() -> None:
    raw = "  کلاسترهای  کوبرنتیز    \n\n\n   تنظیمات   "
    normalized = normalize_search_query(raw)
    assert normalized == "کلاسترهای کوبرنتیز\n\nتنظیمات"


def test_unicode_nfc_normalization() -> None:
    normalizer = PersianNormalizer()
    # NFD decomposition of accented / combined characters
    text_nfd = unicodedata.normalize("NFD", "کتابخانه‌ی ملی")
    cleaned = normalizer.clean_persian_letters(text_nfd)
    assert unicodedata.is_normalized("NFC", cleaned)
