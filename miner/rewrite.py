"""Version C (owner 30/10, FLOW = "rewrite"): the round-0 leaders' style instead of the SFT review conversation.
Every call is a fresh one-turn request to gpt-6-astra with the same instructions (rewrite_prompts.SYSTEM, lumen-kit's
prompt):

    draft    photo + lumen-kit's "Write the complete Three.js module", effort medium (a draft truncated at the output
             cap is redone once)
    check    the Pod renders the judge's 4 front views of the WRAPPED module (lumen_wrap.wrap: centered, largest side
             0.99; 512 px, white, neutral light); not wrappable or rejected -> lumen-kit's repair
             turn (errors + the module + photo -> the complete module) on the draft's model and effort (owner 30/10:
             a repair uses the model and effort the module was written with), <= MAX_REPAIRS
    rewrite  REWRITES times, effort high: golden217's refine turn (photo, the 4 renders, the module) -> a discrepancy
             list + the COMPLETE module. Accepted when the Pod renders it (wrapped); a rejected, truncated or failed
             rewrite is dropped and the next one starts again from the last accepted module (golden217 keeps its gen
             module)
    fallback as versions A/B: a draft call that fails for good -> None; repairs used up -> the last module written,
             unchecked; a failed rewrite call or a budget stop -> the last accepted module
    finish   lumen-kit's post-processing (owner 30/10): lumen_wrap.final (wrap + the flat-object tilt) of the result.
             Not in lumen-kit: the final is rendered once (VIEWS); if that fails, the plain wrap (rendered before) is
             returned instead. An unchecked fallback gets final() without a render, or stays as it is if not wrappable.

The renders the model sees (rewrite turns) are of the wrapped module, as in lumen-kit; no other views yet (owner: later
updates). Budget: the calls share the prompt's credits and calls (pipeline.BUDGET_*), checked before every call from
recorded usage, so a replay decides the same.
"""

from __future__ import annotations

from loguru import logger
from witness_client import AsyncWitnessClient, RenderOptions

from miner import images, lumen_wrap, pipeline, turns
from miner import rewrite_prompts as P

DRAFT_EFFORT, REWRITE_EFFORT = "medium", "high"
REWRITES = 2
MAX_REPAIRS = pipeline.MAX_REPAIRS
CALL_IDS = 100  # call k sends request_ids k*100+1, ... (its resends included)
# The judge's 4 front views in golden217's order: front-left, front-right, from below, from above (= the pick's tiles)
VIEWS = pipeline.TILE_VIEWS
# Owner 29/10: image configurations as in the base miner (detail auto). Both leaders send detail high (a later option).
DETAIL = "auto"


def _estimate(instructions: str, content: list[dict[str, str]]) -> int:
    """Input tokens of a fresh call, generously: instructions and texts at 3 characters a token, 1,500 per image."""
    chars = len(instructions) + sum(len(p.get("text", "")) for p in content)
    return chars // 3 + 1500 * sum(p["type"] == "input_image" for p in content)


class Calls:
    """The prompt's calls so far (versions C, D): each a fresh Conversation with the budget the earlier ones left."""

    def __init__(self, witness: AsyncWitnessClient) -> None:
        self.witness = witness
        self.credits, self.calls, self.k = 0.0, 0, 0

    async def ask(
        self,
        content: list[dict[str, str]],
        effort: str,
        model: str = pipeline.ASTRA,
        instructions: str = P.SYSTEM,
        max_output: int | None = None,
    ) -> tuple[str, str | None]:
        chat = pipeline.Conversation(
            self.witness,
            content,
            model,
            self.k * CALL_IDS,
            pipeline.BUDGET_CREDITS - self.credits,
            pipeline.BUDGET_CALLS - self.calls,
            instructions=instructions,
            effort=effort,
            max_output=max_output,
        )
        # the budget check adds NEXT_TURN_TOKENS back
        chat.context = _estimate(instructions, content) - pipeline.NEXT_TURN_TOKENS
        self.k += 1
        try:
            return await chat.reply()
        finally:
            self.credits += chat.credits
            self.calls += chat.calls


def code_of(text: str) -> str:
    """The module of a reply (fenced or raw, as lumen-kit's prompt asks), ending with one newline."""
    code = turns.extract_code(text).strip()
    return code + "\n" if code else ""


def text_part(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def image_part(url: str) -> dict[str, str]:
    return {"type": "input_image", "image_url": url, "detail": DETAIL}


async def check(module: str, witness: AsyncWitnessClient, views: RenderOptions) -> tuple[bytes | None, list[str]]:
    """pipeline.check of the wrapped module (lumen-kit's _chain: a module that cannot be wrapped is an error)."""
    wrapped = lumen_wrap.wrap(module)
    if wrapped is None:
        return None, [lumen_wrap.NO_EXPORT]
    return await pipeline.check(wrapped, witness, views)


async def finish(module: str, witness: AsyncWitnessClient, checked: bool = True) -> str:
    """lumen-kit's final(): wrap + the flat-object tilt. A checked module's final is rendered once (lumen-kit does
    not); a final that fails falls back to the plain wrap, which rendered before."""
    out = lumen_wrap.final(module)
    if out is None:  # only an unchecked fallback can lack the export line
        return module
    if checked:
        grid, errors = await pipeline.check(out, witness, VIEWS)
        if grid is None:
            logger.warning("final post-pass rejected ({}), returning the plain wrap", errors[:1])
            return lumen_wrap.wrap(module) or module
    return out


async def first_module(
    calls: Calls,
    photo: dict[str, str],
    model: str = pipeline.ASTRA,
    effort: str = DRAFT_EFFORT,
    views: RenderOptions = VIEWS,
) -> tuple[str, bytes | None]:
    """Draft and repairs, all on model at effort (owner 30/10: a repair uses the draft's model and effort) ->
    (the accepted module and its renders of views) or (the last module written or "", None)."""
    code = ""
    for attempt in (1, 2):
        try:
            text, incomplete = await calls.ask([text_part(P.GEN_USER), photo], effort, model)
        except pipeline.ModelFailure as e:
            logger.warning("draft call failed ({}), nothing to return", e)
            return "", None
        code = code_of(text)
        if incomplete == "max_output_tokens" and not turns.looks_like_module(code):
            logger.warning("draft hit max_output_tokens" + (", redoing it" if attempt == 1 else " twice"))
            code = ""
            continue
        break
    tried, repairs = "", 0
    while code:
        if turns.looks_like_module(code):
            tried = code
        grid, errors = await check(code, calls.witness, views)
        if grid is not None:
            logger.info("{} accepted", "draft" if not repairs else f"repair {repairs}")
            return code, grid
        if repairs >= MAX_REPAIRS:
            logger.warning(
                "no valid module after {} repairs ({}), returning the last one unchecked", repairs, errors[:1]
            )
            break
        repairs += 1
        logger.info("check failed ({}), repair {}", errors[:1], repairs)
        try:
            text, _ = await calls.ask([text_part(P.REPAIR_USER % ("\n".join(errors), code)), photo], effort, model)
        except pipeline.ModelFailure as e:
            logger.warning("repair call failed ({}), returning the last module", e)
            break
        code = code_of(text)
    return tried, None


def refine_content(step: int, steps: int, photo: dict[str, str], grid: bytes, module: str) -> list[dict[str, str]]:
    """golden217's refine turn (versions C and D): the photo, the 4 front views (VIEWS) of the module and its code."""
    return [
        text_part(P.REFINE_TEXT.format(step=step, steps=steps)),
        text_part(P.REFERENCE_LABEL),
        photo,
        text_part(P.RENDERS_LABEL),
        *(image_part(url) for url in images.view_data_urls(grid, VIEWS.img_size)),
        text_part(f"{P.CODE_LABEL}\n```javascript\n{module.rstrip()}\n```"),
    ]


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    calls = Calls(witness)
    photo = image_part(images.photo_data_url(image))
    best, grid = await first_module(calls, photo)
    if grid is None:
        return await finish(best, witness, checked=False) if best else None
    for step in range(1, REWRITES + 1):
        content = refine_content(step, REWRITES, photo, grid, best)
        try:
            text, incomplete = await calls.ask(content, REWRITE_EFFORT)
        except pipeline.ModelFailure as e:  # failed for good or a budget stop: the last accepted module
            logger.warning("rewrite {} call failed ({}), keeping the last accepted module", step, e)
            break
        if incomplete:
            logger.info("rewrite {} incomplete ({}), dropped", step, incomplete)
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, witness, VIEWS)
        if new_grid is None:
            logger.info("rewrite {} rejected ({}), dropped", step, errors[:1])
            continue
        logger.info("rewrite {} accepted", step)
        best, grid = cand, new_grid
    return await finish(best, witness)
