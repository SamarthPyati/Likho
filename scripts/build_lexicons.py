"""Build the committed frequency lexicons in likho/data/ from wordfreq.

    python scripts/build_lexicons.py

Run once; outputs are committed. wordfreq data is CC-BY-SA 4.0 — see
likho/data/README.md for attribution.
"""

import re
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

OUT = Path(__file__).resolve().parent.parent / "likho" / "data"
EN_TOP_N = 50_000
LATIN_WORD = re.compile(r"[a-z]+")
DEVANAGARI_WORD = re.compile(r"[ऀ-ॿ‌‍]+")


def write(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("word\tzipf\n")
        f.writelines(f"{w}\t{z:.2f}\n" for w, z in rows)


def main() -> None:
    en = [w for w in top_n_list("en", EN_TOP_N) if LATIN_WORD.fullmatch(w)]
    write(OUT / "en_freq.tsv", [(w, zipf_frequency(w, "en")) for w in en])

    hi = [w for w in top_n_list("hi", 1_000_000) if DEVANAGARI_WORD.fullmatch(w)]
    write(OUT / "hi_freq.tsv", [(w, zipf_frequency(w, "hi")) for w in hi])


if __name__ == "__main__":
    main()
