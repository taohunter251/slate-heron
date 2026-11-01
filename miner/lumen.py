"""Version D (owner 30/10, FLOW = "lumen"): the round-0 leader lumen-kit's pipeline (research-somthing-t/lumen-kit
@3594fd1a sn17gen/pipeline.py) with every call on gpt-6-astra except one Sol draft and each chain's first rewrite
(Sol high). Every call is a fresh one-turn request with lumen-kit's texts (rewrite_prompts.py), except the rewrites,
which send version C's (golden217's) refine turn (owner 30/10):

    chains   the DRAFTS chains run one after the other, each:
             draft (chain 1 gpt-6-astra, chain 2 gpt-6.1-sol, both effort medium)
             -> check: the judge's 4 front views as in version C (rewrite.VIEWS: front-left, front-right, from below,
             from above; 512 px, white, neutral) of the WRAPPED module (lumen_wrap.wrap, as lumen-kit renders)
             -> repairs as in version C (the draft's model and effort, lumen-kit's repair turn, <= MAX_REPAIRS)
             -> one rewrite per REWRITES entry (gpt-6.1-sol high, then gpt-6-astra high): version C's refine turn
             (rewrite.refine_content: "REFINEMENT STEP k of 2", discrepancy list, photo + the 4 front views + the
             module -> the complete module); kept when it renders, else dropped
    pick     with 2+ accepted chain finals: Astra low scores them on the judge's 4 front views (lumen-kit's panel
             prompt and 0-10 penalty rubric, 384 px sheets); the lowest penalty wins; a tie, an unusable reply or a
             failed call -> the first chain
    fallback no accepted module in any chain -> the first chain's last module written (unchecked), else None
    finish   lumen-kit's final() (wrap + flat-object tilt) of the result, as version C (rewrite.finish: the final is
             rendered once, which lumen-kit does not do; a failure falls back to the plain wrap)

Changes from lumen-kit (owner: models and efforts; the rest so that it runs like versions A-C):
- lumen-kit: both drafts Sol medium, both refines Sol high, repairs Sol high (pick Astra low as here). Here the second
  refine is Astra high (owner 30/10: first rewrite Sol high; was Astra medium).
- Rewrites (owner 30/10: "modify D to use C's rewrite draft"): version C's refine text and views instead of
  lumen-kit's REFINE_USER on its 2x2 sheet (front-left, front-right, from above, right side).
- Post-pass as lumen-kit (owner 30/10): renders of the wrapped module, final() on the result; added: the final's render
  check with the plain-wrap fallback.
- A rejected rewrite is dropped and the next one starts from the last accepted module (lumen-kit repairs it, which
  uses up its next refine call).
- Budget: our guard (the next call's worst case must fit the prompt's 5M credits and 60 calls) instead of lumen-kit's
  4.2M cap with an expected 450k per call. Images at detail auto (lumen-kit: high; owner 29/10 base-miner rule).
"""

from __future__ import annotations

import json
import math
import re

from loguru import logger
from witness_client import AsyncWitnessClient, RenderOptions

from miner import images, pipeline, rewrite
from miner import rewrite_prompts as P
from miner.rewrite import Calls, check, code_of, finish, first_module, image_part, refine_content, text_part

# The chains in order: (draft model, draft effort). The first is the pick's fallback and wins ties.
DRAFTS: tuple[tuple[str, str], ...] = ((pipeline.ASTRA, "medium"), (pipeline.MODEL, "medium"))
# The rewrites of every chain in order: (model, effort). Owner 30/10: the first on Sol high (was Astra medium).
REWRITES: tuple[tuple[str, str], ...] = ((pipeline.MODEL, "high"), (pipeline.ASTRA, "high"))
PICK_MODEL, PICK_EFFORT, PICK_MAX_OUTPUT = pipeline.ASTRA, "low", 8192


def _views(pairs: tuple[tuple[float, float], ...], size: int) -> RenderOptions:
    thetas, phis = zip(*pairs, strict=True)
    return RenderOptions(
        thetas=thetas,
        phis=phis,
        img_size=size,
        cam_radius=2.0,
        cam_fov_deg=49.1,
        gap=images.GRID_GAP,
        bg_color="ffffff",
        lighting="neutral",
    )


# lumen-kit views.py: FRONT (pick), (theta, phi) with phi -30 = from above
FRONT_VIEWS = _views(((30.0, 0.0), (330.0, 0.0), (0.0, 15.0), (0.0, -30.0)), 384)


async def chain(calls: Calls, photo: dict[str, str], model: str, effort: str) -> tuple[str, bool]:
    """One chain -> (its last accepted module, True) or (its last module written or "", False)."""
    best, grid = await first_module(calls, photo, model, effort)
    if grid is None:
        return best, False
    for step, (rewrite_model, rewrite_effort) in enumerate(REWRITES, 1):
        content = refine_content(step, len(REWRITES), photo, grid, best)
        try:
            text, incomplete = await calls.ask(content, rewrite_effort, rewrite_model)
        except pipeline.ModelFailure as e:  # failed for good or a budget stop: the last accepted module
            logger.warning("rewrite {} call failed ({}), keeping the last accepted module", step, e)
            break
        if incomplete:
            logger.info("rewrite {} incomplete ({}), dropped", step, incomplete)
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, calls.witness, rewrite.VIEWS)
        if new_grid is None:
            logger.info("rewrite {} rejected ({}), dropped", step, errors[:1])
            continue
        logger.info("rewrite {} accepted", step)
        best, grid = cand, new_grid
    return best, True


def _penalties(text: str, n: int) -> list[float] | None:
    """lumen-kit's parse of the panel reply: the first {...} holding one number per candidate."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        pens = [float(x) for x in json.loads(m.group(0))["penalties"]] if m else []
    except (ValueError, KeyError, TypeError):
        return None
    return pens if len(pens) == n and all(math.isfinite(p) for p in pens) else None


async def pick(calls: Calls, photo: dict[str, str], modules: list[str]) -> str:
    """Astra scores the modules on the judge's front views; the lowest penalty wins, the first on a tie or failure."""
    sheets = []
    for m in modules:
        grid, _ = await check(m, calls.witness, FRONT_VIEWS)
        if grid is None:  # rendered before, so a renderer flake: the pick goes without this module
            logger.warning("pick: a candidate no longer renders, left out")
            continue
        sheets.append((m, images.sheet_data_url(grid, FRONT_VIEWS.img_size)))
    if len(sheets) < 2:
        return sheets[0][0] if sheets else modules[0]
    content = [text_part(P.PANEL_USER % (len(sheets) + 1)), photo, *(image_part(url) for _, url in sheets)]
    try:
        text, _ = await calls.ask(
            content, PICK_EFFORT, PICK_MODEL, instructions=P.PICK_SYSTEM, max_output=PICK_MAX_OUTPUT
        )
    except pipeline.ModelFailure as e:
        logger.warning("pick call failed ({}), keeping the first chain", e)
        return sheets[0][0]
    pens = _penalties(text, len(sheets))
    if pens is None:
        logger.warning("pick reply unusable, keeping the first chain")
        return sheets[0][0]
    best = min(range(len(sheets)), key=lambda i: (pens[i], i))
    logger.info("pick: chain {} of {} (penalties {})", best + 1, len(sheets), pens)
    return sheets[best][0]


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    calls = Calls(witness)
    photo = image_part(images.photo_data_url(image))
    finals = [await chain(calls, photo, model, effort) for model, effort in DRAFTS]
    accepted = [m for m, ok in finals if ok]
    if not accepted:
        tried = next((m for m, _ in finals if m), None)
        return await finish(tried, witness, checked=False) if tried else None
    best = accepted[0] if len(accepted) == 1 else await pick(calls, photo, accepted)
    return await finish(best, witness)
