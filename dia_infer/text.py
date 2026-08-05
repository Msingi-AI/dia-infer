"""Swahili TTS text normalization (owned copy; no temp/ dependency)."""

from __future__ import annotations

import re

UNITS = [
    "sufuri",
    "moja",
    "mbili",
    "tatu",
    "nne",
    "tano",
    "sita",
    "saba",
    "nane",
    "tisa",
]
TENS = {
    1: "kumi",
    2: "ishirini",
    3: "thelathini",
    4: "arobaini",
    5: "hamsini",
    6: "sitini",
    7: "sabini",
    8: "themanini",
    9: "tisini",
}
ABBREVIATIONS = {
    "Dkt.": "Daktari",
    "Bw.": "Bwana",
    "Bi.": "Bibi",
    "Prof.": "Profesa",
    "n.k.": "na kadhalika",
    "k.m.": "kwa mfano",
    "k.v.": "kama vile",
}


def cardinal(n: int) -> str:
    if n < 0:
        return "hasi " + cardinal(-n)
    if n < 10:
        return UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        word = TENS[tens]
        return word if unit == 0 else f"{word} na {UNITS[unit]}"
    if n < 1_000:
        hundreds, rest = divmod(n, 100)
        word = f"mia {UNITS[hundreds]}"
    elif n < 1_000_000:
        thousands, rest = divmod(n, 1_000)
        word = f"elfu {cardinal(thousands)}"
    elif n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        word = f"milioni {cardinal(millions)}"
    else:
        billions, rest = divmod(n, 1_000_000_000)
        word = f"bilioni {cardinal(billions)}"
    if rest == 0:
        return word
    if rest < 10 or (rest < 100 and rest % 10 == 0):
        return f"{word} na {cardinal(rest)}"
    return f"{word} {cardinal(rest)}"


def _verbalize_match(match: re.Match) -> str:
    token = match.group(0).replace(",", "")
    if "." in token:
        whole, frac = token.split(".", 1)
        frac_words = " ".join(UNITS[int(d)] for d in frac if d.isdigit())
        return f"{cardinal(int(whole))} nukta {frac_words}"
    return cardinal(int(token))


def expand_numbers(text: str) -> str:
    return re.sub(r"\d[\d,]*(?:\.\d+)?", _verbalize_match, text)


def expand_abbreviations(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text


def normalize_for_tts(text: str) -> str:
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def format_sw_prompt(text: str) -> str:
    """Normalize and prefix [sw] for msingiai/dia (never [S1]/[S2])."""
    body = normalize_for_tts(text.strip())
    if body.startswith("["):
        return body
    return f"[sw]{body}"
