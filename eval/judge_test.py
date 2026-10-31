import pytest

from eval.judge import FALLBACK_SCORE, judge
from likho.mocks import MockLLM
from likho.schemas import Register


class FixedLLM(MockLLM):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str, **kwargs) -> str:
        return self.reply


def test_parses_scores_with_preamble():
    llm = FixedLLM('Scores:\n{"meaning": 5, "register": 4, "fluency": "3"}')
    score = judge("मूल", "नया", Register.FORMAL, llm=llm)
    assert (score.meaning, score.register, score.fluency, score.parsed) == (5, 4, 3, True)


@pytest.mark.parametrize(
    "reply",
    [
        "Looks good to me!",
        '{"meaning": 5, "register": 4}',  # missing axis
        '{"meaning": 9, "register": 4, "fluency": 4}',  # out of range
        '{"meaning": "great", "register": 4, "fluency": 4}',
        "",
    ],
)
def test_unparseable_reply_gets_fallback_score(reply):
    score = judge("मूल", "नया", Register.NEUTRAL, llm=FixedLLM(reply))
    assert not score.parsed
    assert score.meaning == score.register == score.fluency == FALLBACK_SCORE


def test_model_exception_gets_fallback_score():
    class Broken(MockLLM):
        def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("boom")

    assert not judge("a", "b", Register.CASUAL, llm=Broken()).parsed


def test_mock_judge_parses():
    assert judge("a", "b", Register.CASUAL, llm=MockLLM()).parsed
