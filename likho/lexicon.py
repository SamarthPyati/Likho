"""Word-frequency lexicons committed under likho/data/ (built by scripts/build_lexicons.py).

Frequencies are Zipf values (log10 per billion words) from wordfreq. The
English list contains common romanized Hindi too (`hai`, `ko`), because web
"English" does; the tagger compares against Hindi frequency rather than
trusting membership alone.
"""

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"


def _load(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["word"]: float(row["zipf"]) for row in csv.DictReader(f, delimiter="\t")}


@lru_cache(maxsize=1)
def _english() -> dict[str, float]:
    return _load(DATA / "en_freq.tsv")


@lru_cache(maxsize=1)
def _hindi() -> dict[str, float]:
    return {unicodedata.normalize("NFC", w): z for w, z in _load(DATA / "hi_freq.tsv").items()}


def english_zipf(word: str) -> float:
    """Zipf frequency of a lowercase Latin word in English; 0.0 if unlisted."""
    return _english().get(word.lower(), 0.0)


def hindi_zipf(word: str) -> float:
    """Zipf frequency of a Devanagari word in Hindi; 0.0 if unlisted."""
    return _hindi().get(unicodedata.normalize("NFC", word), 0.0)
