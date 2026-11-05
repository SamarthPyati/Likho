import pytest

from app import gradio_app as ui

TEXT = "kal  ke meeting, Rohit"


def tok(surface, kind, chosen=None, candidates=()):
    return {
        "surface": surface,
        "kind": kind,
        "candidates": list(candidates),
        "chosen": chosen if chosen is not None else surface,
        "chosen_rank": 0 if candidates else None,
    }


TOKENS = [
    tok("kal", "hindi", "कल", ["कल", "काल"]),
    tok("ke", "hindi", "के", ["के", "की"]),
    tok("meeting", "english"),
    tok(",", "punct"),
    tok("Rohit", "named_entity"),
]


def response(**overrides):
    base = {
        "route": {"stages": ["route", "tokenize"], "reason": "Roman Hindi", "source": "llm"},
        "tokens": [dict(t) for t in TOKENS],
        "native_script": "कल  के meeting, Rohit",
        "rewritten": None,
        "english": None,
        "errors": [],
        "timings_ms": {"route": 12.0, "total": 1500.0},
        "text": TEXT,
    }
    return base | overrides


def test_render_keeps_spacing():
    assert ui.render(TEXT, TOKENS) == "कल  के meeting, Rohit"


def test_render_rejects_foreign_tokens():
    with pytest.raises(ValueError):
        ui.render("abc", [tok("xyz", "hindi")])


def test_highlights_map_entries_back_to_tokens():
    entries, owners = ui.highlights(TEXT, TOKENS)
    assert entries == [
        ("कल", "Hindi"),
        ("  ", None),
        ("के", "Hindi"),
        (" ", None),
        ("meeting", "English"),
        (",", None),
        (" ", None),
        ("Rohit", "Name"),
    ]
    assert owners == [0, -1, 1, -1, 2, 3, -1, 4]


def test_route_markdown_shows_source_stages_and_timings():
    md = ui.route_markdown(response())
    assert "Param-1, confirmed by rules" in md
    assert "route (12 ms) → tokenize" in md
    assert "total 1.5 s" in md


def test_errors_markdown():
    assert ui.errors_markdown(response()) == ""
    md = ui.errors_markdown(response(errors=[{"stage": "rewrite", "message": "rejected"}]))
    assert "**rewrite**: rejected" in md


class Click:
    def __init__(self, index):
        self.index = index


def test_clicking_hindi_word_offers_candidates():
    _, owners = ui.highlights(TEXT, TOKENS)
    selected, update = ui.pick_word(response(), owners, Click(0))
    assert selected == 0
    assert update["choices"] == ["कल", "काल"] and "kal" in update["label"]


def test_clicking_word_without_candidates_hides_alternatives():
    _, owners = ui.highlights(TEXT, TOKENS)
    selected, update = ui.pick_word(response(), owners, Click(4))  # "meeting"
    assert selected is None and update["choices"] == []


def test_choosing_alternative_rerenders_native_script():
    resp, native, entries, note = ui.choose_alternative(response(), 0, "काल")
    assert native == "काल  के meeting, Rohit"
    assert resp["tokens"][0]["chosen_rank"] == 1
    assert entries[0] == ("काल", "Hindi")
    assert "Update rewrite" in note
