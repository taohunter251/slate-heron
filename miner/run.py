"""Run a generation function over pending prompts and submit results to the Pod."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import cast

import aiofiles
import httpx
from bittensor_wallet import Wallet
from loguru import logger
from witness_client import AsyncWitnessClient, WitnessConfig

ROOT = Path(__file__).resolve().parents[1]
# seconds between passes over the unfinished prompts, then 300 each: a renderer restart, a full Pod queue or a judge-render
# 500 at the final save often clears within minutes, and a pass without a wait would hit it again (huy-gpt r19_D)
PASS_WAITS = (30, 60, 120, 300)


async def read_image(source: str | bytes, http: httpx.AsyncClient) -> bytes:
    if isinstance(source, bytes):
        return source
    if source.startswith(("https://", "http://")):
        response = await http.get(source)
        response.raise_for_status()
        return response.content
    async with aiofiles.open(Path(source).expanduser(), "rb") as image_file:
        return await image_file.read()


async def run(
    generate: Callable[[bytes, int, AsyncWitnessClient], Awaitable[str | None]],
    prompts: Mapping[str, str | bytes],
    seed: int,
    *,
    workers: int = 8,
    parallel_calls: int = 3,
    attempts: int = 10,
    wallet_name: str = "miner",
    wallet_hotkey: str = "default",
    wallet_path: str = "~/.bittensor/wallets",
    api: bool = False,
) -> None:
    remaining = dict(prompts)
    if not remaining:
        return
    config = cast(WitnessConfig, json.loads((ROOT / "witness.json").read_bytes()))
    hotkey = Wallet(name=wallet_name, hotkey=wallet_hotkey, path=str(Path(wallet_path).expanduser())).hotkey

    async def try_run(last: bool) -> None:
        async with (
            AsyncWitnessClient(
                config,
                hotkey=hotkey,
                openai_api_key=os.environ["OPENAI_API_KEY"] if api else None,
                subscription=None if api else "codex",
                parallel_calls=parallel_calls,
            ) as witness,
            httpx.AsyncClient(timeout=30, follow_redirects=True, trust_env=False) as http,
        ):
            submitted = await witness.get_submitted()
            queue = asyncio.Queue[tuple[str, str | bytes]]()
            for prompt_id, source in remaining.copy().items():
                if prompt_id in submitted:
                    logger.info("{} | already submitted", prompt_id[:12])
                    del remaining[prompt_id]
                else:
                    queue.put_nowait((prompt_id, source))

            async def worker() -> None:
                while not queue.empty():
                    prompt_id, source = queue.get_nowait()
                    try:
                        image = await read_image(source, http)
                        logger.info("{} | generating", prompt_id[:12])

                        result = await generate(image, seed, witness)
                        await witness.submit(image, result)

                        logger.info("{} | submitted", prompt_id[:12])

                        del remaining[prompt_id]
                    except Exception:  # noqa: BLE001
                        logger.exception("Prompt {} failed", prompt_id)

            await asyncio.gather(*(worker() for _ in range(workers)))
            if not remaining or last:
                missing = set(prompts) - set(await witness.get_submitted())
                if missing:
                    logger.error("{} of {} prompts not submitted: {}", len(missing), len(prompts), sorted(missing))
                else:
                    logger.info("all {} prompts submitted", len(prompts))

    for attempt in range(1, attempts + 1):
        logger.info("Attempt {}/{} | {} prompts remaining", attempt, attempts, len(remaining))
        try:
            await try_run(attempt == attempts)
        except Exception:  # noqa: BLE001
            logger.exception("Attempt {}/{} failed", attempt, attempts)
        if not remaining:
            return
        if attempt < attempts:
            wait = PASS_WAITS[min(attempt, len(PASS_WAITS)) - 1]
            logger.warning("{} prompts pending, next pass in {}s: {}", len(remaining), wait, sorted(remaining))
            await asyncio.sleep(wait)

    raise RuntimeError(f"{len(remaining)} prompts remain after {attempts} attempts: {sorted(remaining)}")
