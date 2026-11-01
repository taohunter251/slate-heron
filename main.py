"""Generate Three.js assets from images using a witness Pod."""

import argparse
import asyncio
from pathlib import Path

from witness_client import AsyncWitnessClient

from miner import pipeline

__all__ = ["generate"]


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    return await pipeline.generate(image, seed, witness)


async def main() -> None:
    from miner.run import run  # live runner only; generation and replay never import it

    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompts", type=Path, help="Text file with one image URL per line")
    inputs.add_argument("--image", action="append", help="Image file or HTTP(S) URL; repeat for a small batch")
    parser.add_argument("--seed", type=int, required=True, help="Published round seed")
    parser.add_argument("--parallel-prompts", type=int, default=8, help="Active prompts (default: 8)")
    parser.add_argument("--parallel-calls", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=10, help="Maximum passes over unfinished prompts (default: 10)")
    parser.add_argument("--wallet-name", default="miner")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets")
    parser.add_argument("--api", action="store_true", help="Use OPENAI_API_KEY instead of a local Codex subscription")
    args = parser.parse_args()

    if args.prompts:
        sources = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        sources = args.image

    await run(
        generate,
        {Path(source).stem: source for source in sources},
        args.seed,
        workers=args.parallel_prompts,
        parallel_calls=args.parallel_calls,
        attempts=args.attempts,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
        api=args.api,
    )


if __name__ == "__main__":
    asyncio.run(main())
