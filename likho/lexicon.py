"""Word-frequency lexicons committed under likho/data/ (built by scripts/build_lexicons.py).

Frequencies are Zipf values (log10 per billion words) from wordfreq. The
English list contains common romanized Hindi too (`hai`, `ko`), because web
"English" does; the tagger compares against Hindi frequency rather than
trusting membership alone.
"""

import csv
import math
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


BACKOFF_LOG10 = math.log10(0.4)  # "stupid backoff" penalty for an unseen bigram


@lru_cache(maxsize=1)
def _ngrams() -> tuple[dict[str, int], dict[tuple[str, str], int], int]:
    unigrams: dict[str, int] = {}
    bigrams: dict[tuple[str, str], int] = {}
    total = 0
    with (DATA / "hi_bigrams.tsv").open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("# total_tokens"):
                total = int(line.split("\t")[1])
            if line.startswith(("#", "first\t")):
                continue
            first, second, count = line.rstrip("\n").split("\t")
            if second:
                bigrams[first, second] = int(count)
            else:
                unigrams[first] = int(count)
    return unigrams, bigrams, total


def bigram_log_prob(prev: str, word: str) -> float:
    """log10 P(word | prev) from the IndicCorpV2 sample, with stupid backoff to
    the unigram probability (count 0.5 for unseen words)."""
    unigrams, bigrams, total = _ngrams()
    prev, word = unicodedata.normalize("NFC", prev), unicodedata.normalize("NFC", word)
    pair = bigrams.get((prev, word))
    if pair:
        return math.log10(pair / unigrams[prev])
    return BACKOFF_LOG10 + math.log10(unigrams.get(word, 0.5) / total)
