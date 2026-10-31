"""Build likho/data/hi_bigrams.tsv from a sample of AI4Bharat IndicCorpV2 (Hindi, CC0).

    python -m scripts.build_ngrams

Downloads fixed byte ranges of data/hi-1.txt (26.7 GB; never fetched whole),
counts adjacent Devanagari word pairs, and writes unigram + bigram counts.
Ranges, sizes and cut-offs are constants, so reruns produce the same file.
"""

import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from likho.tokenize import TOKEN_RE

URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-1.txt"
CHUNK_BYTES = 8_000_000
OFFSETS = [0, 5_000_000_000, 10_000_000_000, 15_000_000_000, 20_000_000_000, 25_000_000_000]
MIN_UNIGRAM = 2
MIN_BIGRAM = 3
OUT = Path(__file__).resolve().parent.parent / "likho" / "data" / "hi_bigrams.tsv"


def fetch(offset: int) -> list[str]:
    request = urllib.request.Request(
        URL, headers={"Range": f"bytes={offset}-{offset + CHUNK_BYTES - 1}"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        lines = response.read().decode("utf-8", errors="ignore").splitlines()
    return lines[1:-1]  # first and last lines are cut mid-way by the byte range


def count(lines: list[str]) -> tuple[Counter, Counter]:
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for line in lines:
        prev = None
        for match in TOKEN_RE.finditer(unicodedata.normalize("NFC", line)):
            word = match.group() if match.lastgroup == "devanagari" else None
            if word:
                unigrams[word] += 1
                if prev:
                    bigrams[prev, word] += 1
            prev = word  # punctuation, digits and Latin break the chain
    return unigrams, bigrams


def main() -> None:
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for offset in OFFSETS:
        u, b = count(fetch(offset))
        unigrams.update(u)
        bigrams.update(b)

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# IndicCorpV2 hi-1.txt sample: {len(OFFSETS)} x {CHUNK_BYTES} bytes, CC0\n")
        f.write(f"# total_tokens\t{sum(unigrams.values())}\n")
        f.write("first\tsecond\tcount\n")
        for word, n in sorted(unigrams.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= MIN_UNIGRAM:
                f.write(f"{word}\t\t{n}\n")  # empty second column = unigram row
        for (a, b), n in sorted(bigrams.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= MIN_BIGRAM:
                f.write(f"{a}\t{b}\t{n}\n")


if __name__ == "__main__":
    main()
