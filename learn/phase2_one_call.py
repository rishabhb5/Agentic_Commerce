"""Phase 2: one LLM call, no tools.

Sends your message straight to Claude and prints the reply, the tokens used
and what it cost. Claude has no access to the catalog here, so any products
it mentions are guesses from its training data (names, prices and stock may
be wrong or made up). Phase 3 fixes that by giving it a search tool.

    python -m learn.phase2_one_call "black leggings, size S, under $100"
"""

from __future__ import annotations

import os
import sys

import anthropic
from dotenv import load_dotenv

from shopper.pricing import DEFAULT_MODEL, usage_from_anthropic

SYSTEM = "You are a shopping assistant for Alo Yoga. Recommend 1-3 products with prices."


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv) or "black leggings, size S, under $100"
    model = os.getenv("SHOPPER_MODEL", DEFAULT_MODEL)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": question}],
    )

    print(f"You: {question}\n")
    for block in response.content:
        if block.type == "text":
            print(block.text)
    print(f"\n[{model}] {usage_from_anthropic(response.usage).describe(model)}")
    print("Note: no catalog access, so these products are Claude's guesses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
