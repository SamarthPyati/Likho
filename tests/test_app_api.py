"""The Gradio callbacks against the real API app (in-process, mock models)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app as api_app
from app import gradio_app as ui


@pytest.fixture
def api(monkeypatch):
    with TestClient(api_app) as client:

        def post(url, json, timeout):
            return client.post(url, json=json)

        monkeypatch.setattr(ui.httpx, "post", post)
        yield


def test_submit_and_correct_a_word(api):
    text = "kal ke meeting ke liye Rohit ko bata dena"
    resp, owners, _, native, rewritten, english, entries, route_md, errors_md, _ = ui.submit(
        text, "formal", True, False
    )
    assert native == "कल के meeting के लिए Rohit को बता देना"
    assert english.startswith("[EN] ")
    assert errors_md == ""
    assert "transliterate" in route_md

    first = owners.index(0)
    selected, alternatives = ui.pick_word(resp, owners, type("E", (), {"index": first})())
    other = next(c for c in alternatives["choices"] if c != "कल")
    resp, native, _, _ = ui.choose_alternative(resp, selected, other)
    assert native.startswith(other)

    rewritten, english, _ = ui.update_rewrite(resp, "formal", True)
    assert rewritten == native  # MockLLM rewrite is the identity
    assert english == "[EN] " + native


def test_api_down_is_a_readable_error(monkeypatch):
    def refuse(url, **kw):
        raise ui.httpx.ConnectError("refused")

    monkeypatch.setattr(ui.httpx, "post", refuse)
    with pytest.raises(ui.gr.Error, match="make api"):
        ui.submit("kal", "neutral", False, False)


@pytest.mark.parametrize("demo", ui.DEMOS, ids=lambda d: d.label)
def test_every_demo_runs_cleanly(api, demo):
    index, header, entries, native, rewritten, english, route_md, errors_md = ui.run_demo(demo)
    assert ui.DEMOS[index] is demo
    assert demo.text in header and demo.shows in header
    assert errors_md == ""
    assert "".join(e for e, _ in entries)  # every demo produces tagged words
    assert rewritten
    assert bool(english) == demo.translate


def test_demos_show_what_they_claim(api):
    by_label = {d.label.split(" ", 1)[1]: ui.run_demo(d) for d in ui.DEMOS}
    assert "Rohit" in by_label["Names and English words kept"][3]
    assert "meeting" in by_label["Names and English words kept"][3]
    assert "meeting" not in by_label["English words in Devanagari"][3]
    assert "transliterate" not in by_label["Plain English"][6]


def test_compare_tones_returns_one_rewrite_per_register(api):
    assert len(ui.compare_tones("bhai kal tak report bhej dena")) == len(ui.REGISTERS)
    with pytest.raises(ui.gr.Error):
        ui.compare_tones("  ")


def test_open_in_editor_copies_demo_settings():
    text, register, translate, loanwords, _ = ui.open_in_editor(ui.DEMOS.index(ui.DEMOS[2]))
    assert (text, loanwords) == (ui.DEMOS[2].text, True)
    assert register == "neutral" and translate is False


def test_ui_builds():
    assert ui.build() is not None
