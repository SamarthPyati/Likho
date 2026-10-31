"""FR4: choose among IndicXlit's top-k candidates using sentence context.

    score(c) = α · rank_score(c) + β · unigram(c) + γ · bigram(prev_chosen, c)

- rank_score: −rank in IndicXlit's list (0 for its top-1)
- unigram: Hindi Zipf frequency (wordfreq); unlisted words get UNKNOWN_ZIPF
- bigram: log10 P(c | previous Devanagari word) from the IndicCorpV2 sample,
  with stupid backoff; 0 for every candidate when there is no previous
  Devanagari word, so it cannot change the choice there

Greedy, left to right: `prev_chosen` is the word this pass chose for the
previous token. See docs/ARCHITECTURE.md §5.
"""

from dataclasses import dataclass

from likho.lexicon import bigram_log_prob, hindi_zipf
from likho.logging import get_logger
from likho.schemas import Token, TokenKind

logger = get_logger(__name__)

UNKNOWN_ZIPF = 2.5  # below the 3.01 floor of likho/data/hi_freq.tsv


@dataclass(frozen=True)
class RerankWeights:
    alpha: float  # IndicXlit rank
    beta: float  # unigram frequency
    gamma: float  # bigram with the previous word


# Grid-searched on eval/data/dev_cases.yaml (python -m eval.run_eval --stage rerank
# --split dev --tune), 78 gold-HINDI tokens: top-1 91.0% off → 96.2% for β=2 with
# any γ in [0, 4] (a plateau; at β=1, γ>0 adds one token). γ=0.5 is the plateau
# midpoint, chosen before the test set was run. Never tuned on the test set.
DEFAULT_WEIGHTS = RerankWeights(alpha=1.0, beta=2.0, gamma=0.5)
OFF = RerankWeights(alpha=1.0, beta=0.0, gamma=0.0)  # reproduces IndicXlit's own order


def score(candidate: str, rank: int, prev: str | None, weights: RerankWeights) -> float:
    zipf = hindi_zipf(candidate) or UNKNOWN_ZIPF
    context = bigram_log_prob(prev, candidate) if prev and weights.gamma else 0.0
    return -weights.alpha * rank + weights.beta * zipf + weights.gamma * context


def rerank_tokens(tokens: list[Token], weights: RerankWeights = DEFAULT_WEIGHTS) -> list[Token]:
    """Return copies of `tokens` with `chosen` / `chosen_rank` set by `score`.

    `chosen_rank` is the winning candidate's position in IndicXlit's list, so
    rank > 0 means reranking overrode the model (F8 analysis relies on it).
    """
    out: list[Token] = []
    prev: str | None = None
    overridden = 0
    for token in tokens:
        if token.candidates:
            scores = [score(c, r, prev, weights) for r, c in enumerate(token.candidates)]
            best = max(range(len(scores)), key=lambda r: (scores[r], -r))
            token = token.model_copy(update={"chosen": token.candidates[best], "chosen_rank": best})
            overridden += best > 0
        out.append(token)
        prev = token.chosen if _is_devanagari_output(token) else None
    logger.debug("rerank overrode top-1 for %d of %d tokens", overridden, len(tokens))
    return out


def _is_devanagari_output(token: Token) -> bool:
    if token.kind is TokenKind.DEVANAGARI:
        return True
    return bool(token.candidates) and token.chosen is not None
