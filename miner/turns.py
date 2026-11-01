"""Turn texts and reply parsing of the SFT data-gen pipeline, unchanged (sft-data-gen code/pipeline/turns.py with
PROMPT_FILE=prompts/GEN_PROMPT_compact.md, REVIEW_FILE=prompts/REVIEW_sync_api2.txt, INSTRUCTION default, and
coder.py's _strip_fences / _looks_like_module).

Replies: a ```json list of {old, new} edits is applied to the current code; a ```javascript block or a raw
`export default function generate(THREE)` module replaces it."""

from __future__ import annotations

import json
import re

INSTRUCTION = "Reference photo. Write the module."
REVIEW = "Passed. compare.png above. Reply DONE or json edits."

_JS = re.compile(r"```(?:javascript|js)\s*\n(.*?)```", re.DOTALL)
_JSON = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def repair_text(errors: list[str]) -> str:
    errs = "\n".join(f"- {e}" for e in errors) or "- (none)"
    return f"Check failed:\n{errs}\nReply json edits that fix it."


def strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def looks_like_module(text: str) -> bool:
    return "export default function generate(THREE)" in text and "return" in text


def extract_code(text: str) -> str:
    m = _JS.search(text or "")
    return m.group(1).strip() + "\n" if m else strip_fences(text)


def is_done(text: str) -> bool:
    return "DONE" in (text or "")[:200] and not _JSON.search(text or "")


def apply_reply(code: str, text: str) -> tuple[str, bool]:
    """-> (new_code, changed). JSON edits are applied in order, each to its first exact match; edits whose `old`
    text is missing are skipped. While there is no module yet (an empty or broken draft), an edit with an empty `old`
    is the whole new code: the model answers a BAD_OUTPUT repair that way, and skipping it lost the prompt."""
    m = _JSON.search(text or "")
    if m:
        try:
            edits = json.loads(m.group(1))
        except json.JSONDecodeError:
            edits = []
        new = code
        for e in edits if isinstance(edits, list) else []:
            old, rep = (e.get("old"), e.get("new", "")) if isinstance(e, dict) else (None, None)
            if not isinstance(old, str) or not isinstance(rep, str):
                continue  # a malformed edit (no dict, non-text old/new) is skipped like one whose old text is missing
            if old == "" and not looks_like_module(new):
                new = strip_fences(rep) + "\n"
            elif old and old in new:
                new = new.replace(old, rep, 1)
        return new, new != code
    if _JS.search(text or "") or "export default function generate(THREE)" in (text or ""):
        new = extract_code(text)
        return new, new != code
    return code, False
