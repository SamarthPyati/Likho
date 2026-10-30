from likho.mocks import MockXlitEngine
from likho.schemas import Token, TokenKind
from likho.transliterate import transliterate_tokens


class CountingEngine(MockXlitEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def translit_word(self, word: str, topk: int = 5) -> dict[str, list[str]]:
        self.calls.append(word)
        return super().translit_word(word, topk)


def _tok(surface: str, kind: TokenKind) -> Token:
    return Token(surface=surface, kind=kind)


def test_hindi_tokens_get_candidates_and_provisional_choice():
    [token] = transliterate_tokens([_tok("kal", TokenKind.HINDI)], top_k=5)
    assert token.candidates == ["कल", "काल"]
    assert token.chosen == "कल"
    assert token.chosen_rank == 0


def test_non_hindi_kinds_skip_the_model():
    engine = CountingEngine()
    tokens = [
        _tok("Rohit", TokenKind.NAMED_ENTITY),
        _tok("42", TokenKind.NUMERIC),
        _tok("!", TokenKind.PUNCT),
        _tok("कल", TokenKind.DEVANAGARI),
        _tok("meeting", TokenKind.ENGLISH),
        _tok("", TokenKind.HINDI),
    ]
    out = transliterate_tokens(tokens, top_k=5, engine=engine)
    assert engine.calls == []
    assert [t.chosen for t in out] == ["Rohit", "42", "!", "कल", "meeting", ""]
    assert all(t.candidates == [] for t in out)


def test_loanwords_transliterated_when_opted_in():
    [token] = transliterate_tokens(
        [_tok("meeting", TokenKind.ENGLISH)], top_k=5, transliterate_loanwords=True
    )
    assert token.chosen == "मीटिंग"


def test_repeated_words_call_the_model_once():
    engine = CountingEngine()
    tokens = [_tok(w, TokenKind.HINDI) for w in ["ke", "liye", "Ke"]]
    transliterate_tokens(tokens, top_k=5, engine=engine)
    assert engine.calls == ["ke", "liye"]


def test_top_k_truncates():
    [token] = transliterate_tokens([_tok("hai", TokenKind.HINDI)], top_k=1)
    assert token.candidates == ["है"]


def test_input_tokens_are_not_mutated():
    original = _tok("kal", TokenKind.HINDI)
    transliterate_tokens([original], top_k=5)
    assert original.candidates == [] and original.chosen is None
