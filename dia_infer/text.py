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


def format_clone_prompt(prompt_text: str, generate_text: str) -> str:
    """Voice-clone text: transcript of the reference audio + new text (nari-style concat)."""
    return format_sw_prompt(prompt_text) + format_sw_prompt(generate_text)


def split_for_tts(text: str, max_bytes: int) -> list[str]:
    """Split normalized text into bounded UTF-8 chunks at natural boundaries.

    Dia has fixed text/audio contexts and becomes unstable on long-form input.
    This keeps punctuation when possible and falls back to word boundaries for
    a single long sentence.  It intentionally does not add or rewrite words.
    """
    if max_bytes < 16:
        raise ValueError("max_bytes must be at least 16")
    body = normalize_for_tts(text)
    if not body:
        return []
    if len(body.encode("utf-8")) <= max_bytes:
        return [body]

    clauses = re.split(r"(?<=[.!?;:,])\s+", body)
    pieces: list[str] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause.encode("utf-8")) <= max_bytes:
            pieces.append(clause)
            continue

        current: list[str] = []
        current_bytes = 0
        for word in clause.split():
            word_bytes = len(word.encode("utf-8"))
            extra = word_bytes + (1 if current else 0)
            if current and current_bytes + extra > max_bytes:
                pieces.append(" ".join(current))
                current = [word]
                current_bytes = word_bytes
            else:
                current.append(word)
                current_bytes += extra
        if current:
            pieces.append(" ".join(current))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if current and len(candidate.encode("utf-8")) > max_bytes:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    too_long = [chunk for chunk in chunks if len(chunk.encode("utf-8")) > max_bytes]
    if too_long:
        raise ValueError("text contains a word longer than the segment byte budget")
    return chunks
