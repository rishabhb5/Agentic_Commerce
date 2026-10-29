"""Terminal front end: python -m shopper "black leggings, size S, under $100" """

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from shopper.agent import build_graph, make_llm
from shopper.pricing import DEFAULT_MODEL, Usage, usage_from_langchain
from shopper.tools import get_catalog


def print_update(node: str, update: dict, model: str) -> Usage:
    """--verbose: show what each node did on this pass."""
    usage = Usage()
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage):
            usage = usage.add(usage_from_langchain(msg))
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in msg.tool_calls)
            print(f"  [agent] LLM call → {calls or 'answer'}   ({usage_from_langchain(msg).describe(model)})")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            found = len(json.loads(content)) if content.startswith("[") else 1
            print(f"  [tools] ran {msg.name} → {found} result(s)")
    return usage


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m shopper", description="AI shopping assistant")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--verbose", "-v", action="store_true", help="show each agent/tools pass")
    parser.add_argument("--fake-llm", action="store_true", help="offline rule-based model, no API key")
    args = parser.parse_args(argv)

    if not args.fake_llm and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (add it to .env), or use --fake-llm.", file=sys.stderr)
        return 1
    try:
        catalog = get_catalog()
    except httpx.HTTPError as exc:
        print(f"Couldn't reach the store: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(catalog):,} products from {catalog.store_url}\n")

    if args.fake_llm:
        from shopper.fake_llm import RuleBasedShopper

        llm, model = RuleBasedShopper(), "fake"
    else:
        model = os.getenv("SHOPPER_MODEL", DEFAULT_MODEL)
        llm = make_llm(model)
    graph = build_graph(llm)

    question = " ".join(args.question)
    inputs = {"messages": [("user", question)]}
    total = Usage()
    final = None
    for step in graph.stream(inputs, stream_mode="updates"):
        for node, update in step.items():
            if args.verbose:
                total = total.add(print_update(node, update, model))
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final = msg
    print(f"\n{final.text if final else '(no answer)'}")
    if args.verbose:
        print(f"\n{total.describe(model)}")
    return 0
