"""Gradio frontend: a thin client over the Likho API. Run `make api`, then `make dev`.

Nothing here loads or calls a model, and nothing imports `likho` (CLAUDE.md
rule 1, enforced by tests/test_import_boundaries.py): every model result
arrives over HTTP from api/main.py. Picking a word's alternative (U2) edits
the API's response locally; "Update rewrite" sends the corrected Devanagari
back to the API.
"""

import os
from typing import Any

import gradio as gr
import httpx

API_URL = os.environ.get("LIKHO_API_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 180.0  # the first real-model request on CPU can be slow

EXAMPLES = [
    "kal ke meeting ke liye Rohit ko bata dena",
    "yaar aaj office nahi aa paunga, tabiyat thik nahi hai",
    "मुझे kal तक report भेजनी है",
    "please call me when you reach",
]
KIND_LABEL = {  # TokenKind value → legend label; None = not clickable
    "hindi": "Hindi",
    "english": "English",
    "named_entity": "Name",
    "devanagari": "Devanagari",
    "numeric": None,
    "punct": None,
}
COLORS = {"Hindi": "#f59e0b", "English": "#3b82f6", "Name": "#10b981", "Devanagari": "#a855f7"}

Response = dict[str, Any]  # AssistResponse as JSON


def call_api(path: str, payload: dict) -> Response:
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        hint = "Is `make api` running?"
        raise gr.Error(f"Likho API at {API_URL} failed: {type(exc).__name__}. {hint}") from exc
    return r.json()


def render(text: str, tokens: list[dict]) -> str:
    """`text` with each token replaced by its `chosen` form, spacing kept.
    Tokens are in order and cover every non-space character, so a forward
    search finds each one."""
    parts, pos = [], 0
    for token in tokens:
        start = text.find(token["surface"], pos)
        if start < 0:
            raise ValueError("tokens do not match text")
        parts += [text[pos:start], token["chosen"] or token["surface"]]
        pos = start + len(token["surface"])
    return "".join(parts) + text[pos:]


def highlights(text: str, tokens: list[dict]) -> tuple[list[tuple[str, str | None]], list[int]]:
    """HighlightedText entries for the tokens, plus the token index of each entry
    (-1 for the gaps between tokens) to map a click back to its token."""
    entries: list[tuple[str, str | None]] = []
    owners: list[int] = []
    pos = 0
    for i, token in enumerate(tokens):
        start = text.find(token["surface"], pos)
        if start > pos:
            entries.append((text[pos:start], None))
            owners.append(-1)
        entries.append((token["chosen"] or token["surface"], KIND_LABEL[token["kind"]]))
        owners.append(i)
        pos = start + len(token["surface"])
    return entries, owners


def route_markdown(resp: Response) -> str:
    route = resp["route"]
    source = "Param-1, confirmed by rules" if route["source"] == "llm" else "rules"
    timings = resp["timings_ms"]
    steps = " → ".join(f"{s} ({timings[s]:.0f} ms)" if s in timings else s for s in route["stages"])
    total = f" · total {timings['total'] / 1000:.1f} s" if "total" in timings else ""
    return f"**Route** ({source}): {route['reason']}\n\n{steps}{total}"


def errors_markdown(resp: Response) -> str:
    if not resp["errors"]:
        return ""
    lines = [f"- **{e['stage']}**: {e['message']}" for e in resp["errors"]]
    return "⚠️ Some stages did not complete; earlier results are kept.\n\n" + "\n".join(lines)


def submit(text: str, register: str, translate: bool, loanwords: bool):  # noqa: ANN201
    resp = call_api(
        "/assist",
        {
            "text": text,
            "register": register,
            "translate": translate,
            "transliterate_loanwords": loanwords,
        },
    )
    resp["text"] = text
    entries, owners = highlights(text, resp["tokens"])
    return (
        resp,
        owners,
        None,
        resp["native_script"] or "",
        resp["rewritten"] or "",
        resp["english"] or "",
        entries,
        route_markdown(resp),
        errors_markdown(resp),
        gr.update(choices=[], value=None, visible=False),
    )


def pick_word(resp: Response, owners: list[int], evt: gr.SelectData):  # noqa: ANN201
    """Clicking a word shows its transliteration candidates (U2)."""
    token_index = owners[evt.index] if resp and evt.index < len(owners) else -1
    candidates = resp["tokens"][token_index]["candidates"] if token_index >= 0 else []
    if not candidates:
        return None, gr.update(choices=[], value=None, visible=False)
    token = resp["tokens"][token_index]
    return token_index, gr.update(
        choices=candidates,
        value=token["chosen"],
        label=f"Alternatives for “{token['surface']}”",
        visible=True,
    )


def choose_alternative(resp: Response, token_index: int | None, choice: str | None):  # noqa: ANN201
    """Apply a picked candidate to the native script. No model call."""
    if not resp or token_index is None or choice is None:
        return resp, gr.update(), gr.update(), gr.update()
    token = resp["tokens"][token_index]
    if token["chosen"] == choice:
        return resp, gr.update(), gr.update(), gr.update()
    token["chosen"] = choice
    token["chosen_rank"] = token["candidates"].index(choice)
    resp["native_script"] = render(resp["text"], resp["tokens"])
    entries, _ = highlights(resp["text"], resp["tokens"])
    stale = "✏️ Native script edited. Click **Update rewrite** to redo the rewrite and translation."
    return resp, resp["native_script"], entries, stale


def update_rewrite(resp: Response, register: str, translate: bool):  # noqa: ANN201
    """Rewrite (and translate) the corrected native script via the API."""
    if not resp or not resp.get("native_script"):
        return gr.update(), gr.update(), ""
    again = call_api(
        "/assist",
        {"text": resp["native_script"], "register": register, "translate": translate},
    )
    return again["rewritten"] or "", again["english"] or "", errors_markdown(again)


def build() -> gr.Blocks:
    with gr.Blocks(title="Likho") as demo:
        gr.Markdown(
            "# लिखो · Likho\n"
            "Type Hinglish in Roman letters. Likho writes it in Devanagari, rewrites it "
            "in the tone you pick, and can translate it to English."
        )
        resp_state = gr.State(None)
        owners_state = gr.State([])
        selected_state = gr.State(None)

        text = gr.Textbox(label="Your text", lines=3, placeholder=EXAMPLES[0])
        with gr.Row():
            register = gr.Radio(["casual", "neutral", "formal"], value="neutral", label="Tone")
            translate = gr.Checkbox(label="Translate to English")
            loanwords = gr.Checkbox(label="Write English words in Devanagari too")
        go = gr.Button("Likho", variant="primary")
        gr.Examples(EXAMPLES, inputs=text)

        route_md = gr.Markdown()
        errors_md = gr.Markdown()
        words = gr.HighlightedText(
            label="Words (click a Hindi word to see alternatives)",
            color_map=COLORS,
            combine_adjacent=False,
            show_legend=True,
        )
        alternatives = gr.Radio([], label="Alternatives", visible=False)
        with gr.Row():
            native = gr.Textbox(label="Devanagari", lines=3, buttons=["copy"])
            rewritten = gr.Textbox(label="Rewritten", lines=3, buttons=["copy"])
            english = gr.Textbox(label="English", lines=3, buttons=["copy"])
        update = gr.Button("Update rewrite", size="sm")

        outputs = [
            resp_state,
            owners_state,
            selected_state,
            native,
            rewritten,
            english,
            words,
            route_md,
            errors_md,
            alternatives,
        ]
        inputs = [text, register, translate, loanwords]
        go.click(submit, inputs, outputs)
        text.submit(submit, inputs, outputs)
        words.select(pick_word, [resp_state, owners_state], [selected_state, alternatives])
        alternatives.input(
            choose_alternative,
            [resp_state, selected_state, alternatives],
            [resp_state, native, words, errors_md],
        )
        update.click(
            update_rewrite, [resp_state, register, translate], [rewritten, english, errors_md]
        )
    return demo


if __name__ == "__main__":
    build().launch(server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
