from likho.lexicon import bigram_log_prob
from likho.rerank import DEFAULT_WEIGHTS, OFF, RerankWeights, rerank_tokens
from likho.schemas import Token, TokenKind

H, E, D = TokenKind.HINDI, TokenKind.ENGLISH, TokenKind.DEVANAGARI


def _cand(surface: str, *candidates: str) -> Token:
    return Token(surface=surface, kind=H, candidates=list(candidates), chosen=candidates[0])


def test_off_reproduces_model_order():
    [token] = rerank_tokens([_cand("me", "मे", "में")], OFF)
    assert (token.chosen, token.chosen_rank) == ("मे", 0)


def test_unigram_frequency_overrides_model_rank():
    [token] = rerank_tokens([_cand("me", "मे", "में")], DEFAULT_WEIGHTS)
    assert (token.chosen, token.chosen_rank) == ("में", 1)


def test_bigram_uses_previous_chosen_word():
    context_only = RerankWeights(alpha=0.1, beta=0.0, gamma=1.0)
    tokens = [_cand("ke", "के"), _cand("liye", "लिये", "लिए")]
    out = rerank_tokens(tokens, context_only)
    assert out[1].chosen == "लिए"


def test_devanagari_input_counts_as_previous_word():
    context_only = RerankWeights(alpha=0.1, beta=0.0, gamma=1.0)
    tokens = [Token(surface="के", kind=D, chosen="के"), _cand("liye", "लिये", "लिए")]
    assert rerank_tokens(tokens, context_only)[1].chosen == "लिए"


def test_english_token_breaks_the_context_chain():
    context_only = RerankWeights(alpha=0.1, beta=0.0, gamma=1.0)
    tokens = [_cand("ke", "के"), Token(surface="meeting", kind=E, chosen="meeting")]
    tokens.append(_cand("liye", "लिये", "लिए"))
    # No Devanagari word directly before `liye`: the bigram term is 0 for every
    # candidate, so only the rank term decides and the model's top-1 stays.
    assert rerank_tokens(tokens, context_only)[2].chosen == "लिये"


def test_tokens_without_candidates_are_untouched():
    english = Token(surface="meeting", kind=E, chosen="meeting")
    assert rerank_tokens([english]) == [english]


def test_input_not_mutated():
    token = _cand("me", "मे", "में")
    rerank_tokens([token])
    assert token.chosen == "मे"


def test_bigram_seen_beats_unseen():
    assert bigram_log_prob("के", "लिए") > bigram_log_prob("के", "लिये")
    assert bigram_log_prob("के", "लिए") > bigram_log_prob("के", "क़ख़ग़")
