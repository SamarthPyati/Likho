import pytest

from likho import pipeline
from likho.config import settings
from likho.errors import StageFailure
from likho.pipeline import add_translation, assist, assist_untranslated, disabled_stages
from likho.schemas import AssistRequest, Register, Stage, TokenKind

EXAMPLE = "kal ke meeting ke liye Rohit ko bata dena"


def stage_errors(response) -> dict[Stage, str]:
    return {e.stage: e.message for e in response.errors}


def test_example_sentence_end_to_end():
    r = assist(AssistRequest(text=EXAMPLE))
    assert r.native_script == "कल के meeting के लिए Rohit को बता देना"
    assert r.rewritten == r.native_script  # MockLLM rewrite is the identity
    assert r.english is None
    assert r.errors == []
    kinds = {t.surface: t.kind for t in r.tokens}
    assert kinds["meeting"] is TokenKind.ENGLISH
    assert kinds["Rohit"] is TokenKind.NAMED_ENTITY
    assert r.route.stages == [
        Stage.ROUTE,
        Stage.TOKENIZE,
        Stage.TRANSLITERATE,
        Stage.RERANK,
        Stage.REWRITE,
    ]


def test_timings_cover_every_stage_that_ran():
    r = assist(AssistRequest(text=EXAMPLE, translate=True))
    assert set(r.timings_ms) == {s.value for s in r.route.stages} | {"total"}
    assert all(ms >= 0 for ms in r.timings_ms.values())


def test_translate_appends_stage_and_uses_rewrite():
    r = assist(AssistRequest(text=EXAMPLE, translate=True))
    assert r.route.stages[-1] is Stage.TRANSLATE
    assert r.english == "[EN] " + r.rewritten


def test_candidates_are_returned_for_alternatives_ui():
    r = assist(AssistRequest(text=EXAMPLE, top_k=2))
    kal = next(t for t in r.tokens if t.surface == "kal")
    assert kal.candidates == ["कल", "काल"]
    assert kal.chosen == "कल"


def test_loanword_toggle():
    r = assist(AssistRequest(text=EXAMPLE, transliterate_loanwords=True))
    assert "मीटिंग" in r.native_script
    assert "Rohit" in r.native_script  # names stay Roman either way


def test_devanagari_input_skips_transliteration():
    r = assist(AssistRequest(text="मुझे कल जाना है", translate=True))
    assert Stage.TRANSLITERATE not in r.route.stages
    assert r.native_script == "मुझे कल जाना है"
    assert r.english == "[EN] मुझे कल जाना है"


def test_english_input_is_rewritten_not_translated():
    r = assist(AssistRequest(text="please call me when you reach", translate=True))
    assert r.native_script is None
    assert r.rewritten == "please call me when you reach"
    assert r.english == r.rewritten
    assert Stage.TRANSLATE.value not in r.timings_ms  # no model call


def test_empty_input():
    r = assist(AssistRequest(text="   ", translate=True))
    assert r.tokens == []
    assert r.native_script is None and r.rewritten is None and r.english is None
    assert r.errors == []


def test_failed_rewrite_keeps_transliteration(monkeypatch):
    """FR7: the contract test from docs/ARCHITECTURE.md §3."""

    def broken(text, register):
        raise StageFailure(Stage.REWRITE, "rewrite rejected: length ratio 3.00")

    monkeypatch.setattr(pipeline, "rewrite", broken)
    r = assist(AssistRequest(text=EXAMPLE, register=Register.FORMAL, translate=True))
    assert r.native_script == "कल के meeting के लिए Rohit को बता देना"
    assert r.rewritten is None
    assert stage_errors(r) == {Stage.REWRITE: "rewrite rejected: length ratio 3.00"}
    assert r.english == "[EN] " + r.native_script  # translation falls back to native script


def test_unexpected_exception_is_contained(monkeypatch):
    def buggy(text):
        raise KeyError("oops")

    monkeypatch.setattr(pipeline, "translate", buggy)
    r = assist(AssistRequest(text=EXAMPLE, translate=True))
    assert r.rewritten is not None
    assert stage_errors(r) == {Stage.TRANSLATE: "unexpected error: KeyError"}


def test_failed_rerank_falls_back_to_model_top1(monkeypatch):
    def broken(tokens):
        raise RuntimeError

    monkeypatch.setattr(pipeline, "rerank_tokens", broken)
    r = assist(AssistRequest(text=EXAMPLE))
    assert r.native_script == "कल के meeting के लिए Rohit को बता देना"
    assert set(stage_errors(r)) == {Stage.RERANK}


def test_failed_transliteration_skips_rewrite(monkeypatch):
    def broken(*args, **kwargs):
        raise StageFailure(Stage.TRANSLITERATE, "model error: RuntimeError")

    monkeypatch.setattr(pipeline, "transliterate_tokens", broken)
    r = assist(AssistRequest(text=EXAMPLE, translate=True))
    assert r.native_script is None and r.rewritten is None and r.english is None
    assert set(stage_errors(r)) == {Stage.TRANSLITERATE, Stage.REWRITE, Stage.TRANSLATE}
    assert r.tokens  # the tagging is still returned


def test_failed_routing_still_returns_a_response(monkeypatch):
    def broken(text):
        raise RuntimeError

    monkeypatch.setattr(pipeline, "route", broken)
    r = assist(AssistRequest(text=EXAMPLE))
    assert r.route.stages == [Stage.ROUTE]
    assert set(stage_errors(r)) == {Stage.ROUTE}


def test_disable_stages_env_flag(monkeypatch):
    monkeypatch.setattr(settings, "disable_stages", " Rewrite , translate")
    assert disabled_stages() == {Stage.REWRITE, Stage.TRANSLATE}
    r = assist(AssistRequest(text=EXAMPLE, translate=True))
    assert r.native_script == "कल के meeting के लिए Rohit को बता देना"
    assert r.rewritten is None and r.english is None
    assert stage_errors(r) == {
        Stage.REWRITE: "disabled by DISABLE_STAGES",
        Stage.TRANSLATE: "disabled by DISABLE_STAGES",
    }


def test_unknown_disabled_stage_is_an_error(monkeypatch):
    monkeypatch.setattr(settings, "disable_stages", "rewrit")
    with pytest.raises(ValueError):
        disabled_stages()


def test_user_text_is_not_logged(caplog):
    caplog.set_level("DEBUG", logger="likho")
    assist(AssistRequest(text=EXAMPLE, translate=True))
    assert "meeting" not in caplog.text and "Rohit" not in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        "kal ke meeting ke liye Rohit ko bata dena",
        "मुझे kal तक report भेजनी है",
        "please call me",
        "",
    ],
)
def test_two_halves_equal_one_assist_call(text):
    request = AssistRequest(text=text, translate=True)
    whole = assist(request)
    halves = add_translation(text, assist_untranslated(request))
    ignore = {"timings_ms"}
    assert halves.model_dump(exclude=ignore) == whole.model_dump(exclude=ignore)
    assert set(halves.timings_ms) == set(whole.timings_ms)
