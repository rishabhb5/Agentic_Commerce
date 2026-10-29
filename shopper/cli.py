"""Terminal chat with the shopping agent.

    python -m shopper                      # chat (You: prompt)
    python -m shopper "black leggings, S"  # one question, then exit
    python -m shopper --verbose            # show each agent/tools pass + cost
    python -m shopper --fake-llm           # offline, no API key

In chat, /new starts a fresh conversation and /quit (or Ctrl-D) exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from shopper.agent import build_graph, make_llm, new_turn
from shopper.pricing import DEFAULT_MODEL, Usage, usage_from_langchain
from shopper.tools import get_catalog


def print_update(node: str, update: dict, model: str) -> None:
    """--verbose: show what each node did on this pass."""
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage):
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in msg.tool_calls)
            print(f"  [agent] LLM call → {calls or 'answer'}   ({usage_from_langchain(msg).describe(model)})")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            result = f"{len(json.loads(content))} result(s)" if content.startswith("[") else content[:100]
            print(f"  [tools] ran {msg.name} → {result}")


def ask(graph, text: str, thread_id: str, model: str, verbose: bool) -> tuple[str, list[dict], Usage]:
    """Run one user message through the graph. Earlier turns come from the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    usage, answer = Usage(), "(no answer)"
    for step in graph.stream(new_turn(text), config, stream_mode="updates"):
        for node, update in step.items():
            if verbose:
                print_update(node, update, model)
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    usage = usage.add(usage_from_langchain(msg))
                    if not msg.tool_calls:
                        answer = msg.text
    recommendations = graph.get_state(config).values.get("recommendations", [])
    return answer, recommendations, usage


def product_panel(rec: dict) -> Panel:
    low = rec["price"][0] if isinstance(rec["price"], list) else rec["price"]
    price = f"[bold]${low:,.2f}[/bold]"
    if isinstance(rec["price"], list):
        price += f"–${rec['price'][1]:,.2f}"
    if rec.get("was"):
        price += f"  [strike dim]${rec['was']:,.2f}[/strike dim]  [green]sale[/green]"
    details = " · ".join(x for x in [rec.get("fabric"), rec.get("fit"), ", ".join(rec.get("activities") or [])] if x)
    lines = [
        price + (f"   [dim]{escape(details)}[/dim]" if details else ""),
        f"Colours: {escape(', '.join(rec.get('colors') or []) or '-')}",
        f"Sizes in stock: {escape(', '.join(rec.get('sizes_in_stock') or []) or '-')}",
        f"[italic]{escape(rec.get('reason', ''))}[/italic]",
        f"[link={rec['url']}]{escape(rec['url'])}[/link]",
    ]
    return Panel("\n".join(lines), title=f"[bold]{escape(rec['title'])}[/bold]", title_align="left", expand=False)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m shopper", description="AI shopping assistant")
    parser.add_argument("question", nargs="*", help="ask one question and exit")
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
    print(f"Loaded {len(catalog):,} products from {catalog.store_url}")

    if args.fake_llm:
        from shopper.fake_llm import RuleBasedShopper

        llm, model = RuleBasedShopper(), "fake"
    else:
        model = os.getenv("SHOPPER_MODEL", DEFAULT_MODEL)
        llm = make_llm(model)
    graph = build_graph(llm)
    thread_id = uuid.uuid4().hex

    console = Console()

    def turn(text: str) -> None:
        answer, recommendations, usage = ask(graph, text, thread_id, model, args.verbose)
        console.print(f"\n[bold cyan]Shopper:[/bold cyan] {escape(answer)}\n")
        for rec in recommendations:
            console.print(product_panel(rec))
        if args.verbose:
            console.print(f"[dim]({usage.describe(model)})[/dim]")
        console.print()

    if args.question:
        turn(" ".join(args.question))
        return 0

    print("Ask for something (/new = new conversation, /quit = exit)\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/new":
            thread_id = uuid.uuid4().hex
            print("Started a new conversation.\n")
            continue
        turn(text)
