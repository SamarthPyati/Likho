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
