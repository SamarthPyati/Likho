"""Build eval/data/aksharantar_sample.tsv from the Aksharantar Hindi test split.

Run once; the output is committed and never resampled (docs/EVALUATION.md §1.1).

    python -m eval.sample_aksharantar path/to/hin_test.json

hin_test.json comes from hin.zip at https://huggingface.co/datasets/ai4bharat/Aksharantar.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 3235
TOTAL = 50
OUT = Path(__file__).parent / "data" / "aksharantar_sample.tsv"

# Aksharantar's own `source` labels for the Hindi test split.
CATEGORIES = {
    "Dakshina": "dakshina",  # general vocabulary from the Dakshina dataset
    "AK-Freq": "ak_freq",  # most frequent words
    "AK-NEI": "ak_nei",  # Indian named entities
    "AK-NEF": "ak_nef",  # foreign named entities
}


def sample(test_json: Path) -> list[tuple[str, str, str]]:
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with test_json.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_category[CATEGORIES[row["source"]]].append((row["english word"], row["native word"]))

    rng = random.Random(SEED)
    names = sorted(by_category)
    per_category = [TOTAL // len(names) + (i < TOTAL % len(names)) for i in range(len(names))]
    rows = []
    for name, n in zip(names, per_category, strict=True):
        rows.extend((roman, native, name) for roman, native in rng.sample(by_category[name], n))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test_json", type=Path)
    args = parser.parse_args()

    rows = sample(args.test_json)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["roman", "devanagari", "category"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
