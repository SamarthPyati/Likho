# likho/data

| File | Contents | Built by |
|---|---|---|
| `en_freq.tsv` | Top ~48k alphabetic English words with Zipf frequency | `scripts/build_lexicons.py` |
| `hi_freq.tsv` | ~24k Devanagari Hindi words with Zipf frequency | `scripts/build_lexicons.py` |
| `hi_bigrams.tsv` | Unigram (count ≥ 2) and bigram (count ≥ 3) counts from a 48 MB sample of IndicCorpV2 Hindi | `python -m scripts.build_ngrams` |

The two `*_freq.tsv` files are derived from [wordfreq](https://github.com/rspeer/wordfreq) 3.1.1
(Robyn Speer), whose word lists are licensed **CC-BY-SA 4.0**. These derived
files are redistributed under the same licence.

Regenerate only by rerunning the script, and commit the result.

`hi_bigrams.tsv` is derived from [IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
(AI4Bharat), released under **CC0**: six fixed 8 MB byte ranges of
`data/hi-1.txt`, so the sample is reproducible without downloading 26 GB.
