"""Gradio frontend: a thin client over the Likho API. Run `make api`, then `make dev`.

Nothing here loads or calls a model, and nothing imports `likho` (CLAUDE.md
rule 1, enforced by tests/test_import_boundaries.py): every model result
arrives over HTTP from api/main.py. Picking a word's alternative (U2) edits
the API's response locally; "Update rewrite" sends the corrected Devanagari
back to the API.
"""

import os
from dataclasses import dataclass
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
REGISTERS = ["casual", "neutral", "formal"]


@dataclass(frozen=True)
class Demo:
    """One showcase button on the Demo tab: a request plus what it is meant to show."""

    label: str
    text: str
    shows: str
    register: str = "neutral"
    translate: bool = False
    loanwords: bool = False


DEMOS = [
    Demo(
        "🔤 Roman → Devanagari",
        "yaar aaj office nahi aa paunga, tabiyat thik nahi hai",
        "Hindi words typed in Roman letters are written in Devanagari. "
        "*office* is an English word, so it stays as typed.",
    ),
    Demo(
        "👤 Names and English words kept",
        "kal ke meeting ke liye Rohit ko bata dena",
        "The tagger keeps the name *Rohit* and the English word *meeting* in Roman; "
        "only the Hindi words are converted.",
    ),
    Demo(
        "🅰️ English words in Devanagari",
        "kal ke meeting ke liye Rohit ko bata dena",
        "Same sentence with the loanword switch on: English words are written "
        "in Devanagari too (मीटिंग). Names are still kept.",
        loanwords=True,
    ),
    Demo(
        "🎩 Formal rewrite",
        "bhai kal tak report bhej dena please",
        "Param-1 rewrites the Devanagari in a formal register. A rule check rejects "
        "rewrites that change the script or drop content, keeping the original instead.",
        register="formal",
    ),
    Demo(
        "🌐 Translate to English",
        "mujhe kal tak report bhejni hai",
        "After the rewrite, IndicTrans2 translates the Devanagari to English.",
        translate=True,
    ),
    Demo(
        "🔀 Mixed script",
        "मुझे kal तक report भेजनी है",
        "Input that is already partly Devanagari. The router sees the mix and "
        "converts only the Roman Hindi words.",
    ),
    Demo(
        "🇬🇧 Plain English",
        "please call me when you reach",
        "The router sees English and skips transliteration: there is nothing to convert. "
        "Compare the stage list with the other examples.",
    ),
]


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
        no_alternatives(),
    )


ALTERNATIVES_LABEL = "Alternatives"
ALTERNATIVES_HINT = "Click a Hindi word above to see its other spellings."


def no_alternatives():  # noqa: ANN201
    """The alternatives list, emptied. It is always shown rather than hidden: in
    Gradio 6.17 a Radio created with visible=False did not appear when an update
    set visible=True (found in a browser test; the callback itself was fine)."""
    return gr.update(choices=[], value=None, label=ALTERNATIVES_LABEL, info=ALTERNATIVES_HINT)


def pick_word(resp: Response, owners: list[int], evt: gr.SelectData):  # noqa: ANN201
    """Clicking a word shows its transliteration candidates (U2)."""
    token_index = owners[evt.index] if resp and evt.index < len(owners) else -1
    candidates = resp["tokens"][token_index]["candidates"] if token_index >= 0 else []
    if not candidates:
        return None, no_alternatives()
    token = resp["tokens"][token_index]
    return token_index, gr.update(
        choices=candidates,
        value=token["chosen"],
        label=f"Alternatives for “{token['surface']}”",
        info="Pick one to change the Devanagari, then click Update rewrite.",
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


def run_demo(demo: Demo):  # noqa: ANN201
    """Run one showcase through the API and describe what it shows."""
    resp = call_api(
        "/assist",
        {
            "text": demo.text,
            "register": demo.register,
            "translate": demo.translate,
            "transliterate_loanwords": demo.loanwords,
        },
    )
    switches = [f"tone **{demo.register}**"]
    if demo.translate:
        switches.append("translate **on**")
    if demo.loanwords:
        switches.append("English words in Devanagari **on**")
    header = f"### {demo.label}\n\n> {demo.text}\n\n{', '.join(switches)}\n\n{demo.shows}"
    entries, _ = highlights(demo.text, resp["tokens"])
    return (
        DEMOS.index(demo),
        header,
        entries,
        resp["native_script"] or "",
        resp["rewritten"] or "",
        resp["english"] or "",
        route_markdown(resp),
        errors_markdown(resp),
    )


def compare_tones(text: str):  # noqa: ANN201
    """The same text rewritten in every register, one API call each."""
    if not text.strip():
        raise gr.Error("Type something to compare, or run a demo first.")
    rewrites = []
    for register in REGISTERS:
        resp = call_api("/assist", {"text": text, "register": register})
        rewrites.append(resp["rewritten"] or resp["native_script"] or "")
    return tuple(rewrites)


def open_in_editor(index: int | None):  # noqa: ANN201
    """Copy the last demo's settings to the Write tab and switch to it."""
    if index is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    demo = DEMOS[index]
    return demo.text, demo.register, demo.translate, demo.loanwords, gr.Tabs(selected="write")


def _demo_runner(demo: Demo):  # noqa: ANN202
    def run():  # noqa: ANN202
        return run_demo(demo)

    return run


def build_demo_tab(text, register, translate, loanwords, tabs) -> None:  # noqa: ANN001
    gr.Markdown(
        "Each button runs one example through the full pipeline and explains what it "
        "shows. Use **Open in editor** to change it and try your own variations."
    )
    demo_state = gr.State(None)
    with gr.Row():
        buttons = [gr.Button(d.label, size="sm") for d in DEMOS]
    about = gr.Markdown("*Pick a demo above.*")
    words = gr.HighlightedText(
        label="Words by kind", color_map=COLORS, combine_adjacent=False, show_legend=True
    )
    with gr.Row():
        native = gr.Textbox(label="Devanagari", lines=2, buttons=["copy"])
        rewritten = gr.Textbox(label="Rewritten", lines=2, buttons=["copy"])
        english = gr.Textbox(label="English", lines=2, buttons=["copy"])
    route_md = gr.Markdown()
    errors_md = gr.Markdown()
    edit = gr.Button("✏️ Open in editor", size="sm")

    outputs = [demo_state, about, words, native, rewritten, english, route_md, errors_md]
    for button, demo in zip(buttons, DEMOS, strict=True):
        button.click(_demo_runner(demo), None, outputs)
    edit.click(open_in_editor, demo_state, [text, register, translate, loanwords, tabs])

    gr.Markdown("### 🎚️ Compare tones\nThe same sentence rewritten casual, neutral and formal.")
    with gr.Row():
        compare_text = gr.Textbox(
            label="Text", value="bhai kal tak report bhej dena please", scale=4
        )
        compare = gr.Button("Compare tones", variant="primary", scale=1)
    with gr.Row():
        tone_boxes = [gr.Textbox(label=r.capitalize(), lines=2) for r in REGISTERS]
    compare.click(compare_tones, compare_text, tone_boxes)


def build() -> gr.Blocks:
    with gr.Blocks(title="Likho") as demo:
        gr.Markdown(
            "# लिखो · Likho\n"
            "Type Hinglish in Roman letters. Likho writes it in Devanagari, rewrites it "
            "in the tone you pick, and can translate it to English."
        )
        with gr.Tabs() as tabs:
            with gr.Tab("Write", id="write"):
                text, register, translate, loanwords = build_write_tab()
            with gr.Tab("Demo", id="demo"):
                build_demo_tab(text, register, translate, loanwords, tabs)
    return demo


def build_write_tab():  # noqa: ANN201
    """The main editor. Returns its input controls so the Demo tab can fill them."""
    resp_state = gr.State(None)
    owners_state = gr.State([])
    selected_state = gr.State(None)

    text = gr.Textbox(label="Your text", lines=3, placeholder=EXAMPLES[0])
    with gr.Row():
        register = gr.Radio(REGISTERS, value="neutral", label="Tone")
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
    alternatives = gr.Radio([], label=ALTERNATIVES_LABEL, info=ALTERNATIVES_HINT)
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
    update.click(update_rewrite, [resp_state, register, translate], [rewritten, english, errors_md])
    return text, register, translate, loanwords


if __name__ == "__main__":
    build().launch(server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
