"""Phase 3: tool use, with the loop written out by hand.

Claude gets one tool, `search_products`. Each pass:
  1. send the conversation to Claude
  2. if it asked for a tool, print the request, run catalog.search(), and
     append the result to the conversation
  3. repeat until Claude answers in plain text

This is exactly the loop LangGraph runs for us in phase 4 (agent node →
tools node → back to agent).

    python -m learn.phase3_tools_by_hand "black leggings, size S, under $100"
"""

from __future__ import annotations

import json
import os
import sys

import anthropic
from dotenv import load_dotenv

from shopper.catalog import Catalog, product_summary
from shopper.pricing import DEFAULT_MODEL, Usage, usage_from_anthropic

SYSTEM = (
    "You are a shopping assistant for Alo Yoga. Use search_products to find real "
    "products, then recommend 1-3 of them with price and a one-line reason. "
    "Only mention products the tool returned."
)

SEARCH_TOOL = {
    "name": "search_products",
    "description": "Search the store catalog by keywords and optional filters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": 'keywords, e.g. "leggings"'},
            "department": {"type": "string", "description": "Women, Men, Accessories"},
            "max_price": {"type": "number"},
            "color": {"type": "string"},
            "size": {"type": "string", "description": "e.g. XS, S, M, L"},
        },
        "required": ["query"],
    },
}
MAX_PASSES = 5


def run_tool(catalog: Catalog, name: str, args: dict) -> str:
    if name != "search_products":
        return f"Unknown tool {name}"
    results = catalog.search(**args)
    return json.dumps([product_summary(p) for p in results])


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv) or "black leggings, size S, under $100"
    model = os.getenv("SHOPPER_MODEL", DEFAULT_MODEL)

    catalog = Catalog().load()
    print(f"Loaded {len(catalog):,} products from {catalog.store_url}\n")
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    total = Usage()

    for n in range(1, MAX_PASSES + 1):
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM,
            tools=[SEARCH_TOOL],
            output_config={"effort": "low"},
            messages=messages,
        )
        usage = usage_from_anthropic(response.usage)
        total = total.add(usage)
        print(f"── pass {n}: LLM call ({usage.describe(model)})")

        # Keep the whole assistant turn (thinking + text + tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            print("   Claude answered without a tool call → done\n")
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        results = []
        for call in tool_uses:
            print(f"   Claude asked for: {call.name}({json.dumps(call.input)})")
            output = run_tool(catalog, call.name, call.input)
            print(f"   ran it → {len(json.loads(output))} products sent back")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        # All results for this turn go back in one user message.
        messages.append({"role": "user", "content": results})
    else:
        print(f"Stopped after {MAX_PASSES} passes.")

    print(f"\nTotal: {total.describe(model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
