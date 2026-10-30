from likho.schemas import (
    AssistRequest,
    AssistResponse,
    Register,
    Route,
    Stage,
    StageError,
    Token,
    TokenKind,
)


def test_request_defaults():
    req = AssistRequest(text="kal milte hain")
    assert req.register is Register.NEUTRAL
    assert req.translate is False
    assert req.transliterate_loanwords is False
    assert req.top_k == 5


def test_response_round_trips_through_json():
    resp = AssistResponse(
        route=Route(stages=[Stage.ROUTE, Stage.TRANSLITERATE], reason="roman", source="llm"),
        tokens=[Token(surface="kal", kind=TokenKind.HINDI, candidates=["कल", "काल"], chosen="कल")],
        native_script="कल",
        errors=[StageError(stage=Stage.REWRITE, message="length ratio 2.1")],
    )
    again = AssistResponse.model_validate_json(resp.model_dump_json())
    assert again == resp
    assert again.rewritten is None  # FR7: failed rewrite leaves native_script intact


def test_enums_serialise_as_lowercase_values():
    token = Token(surface="Rohit", kind=TokenKind.NAMED_ENTITY)
    assert token.model_dump(mode="json")["kind"] == "named_entity"
