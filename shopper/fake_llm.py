"""Stand-ins for Claude, so the graph runs without an API key.

- `ScriptedChatModel` replays a fixed list of replies (used by tests).
- `RuleBasedShopper` reads the user's message with a few regexes and calls the
  tools the way Claude would. It is dumb on purpose; `python -m shopper
  --fake-llm` uses it to demo the loop offline.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _FakeBase(BaseChatModel):
    tool_names: list[str] = []

    def bind_tools(self, tools, **kwargs) -> "_FakeBase":
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        raise NotImplementedError


class ScriptedChatModel(_FakeBase):
    """Returns `responses` in order, one per call."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []  # what the model was sent each time

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _reply(self, messages):
        self.calls.append(list(messages))
        return self.responses[len(self.calls) - 1]


def tool_call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


_SIZE_RE = re.compile(r"\bsize\s+([a-z0-9]+)\b", re.I)
_PRICE_RE = re.compile(r"(?:under|below|less than|max|<)\s*\$?\s*(\d+)", re.I)
_COLORS = "black white navy blue green red pink grey gray beige brown purple".split()
_CATEGORIES = set("leggings legging bras bra hoodie hoodies shorts jackets jacket tanks tank joggers jogger pants sweatpants tees skirts dresses socks bags hats".split())
_DEPARTMENTS = {"women": "Women", "womens": "Women", "men": "Men", "mens": "Men"}
_STOPWORDS = set("a an the i me my for some something in of to under below with and or size want need looking any show".split())


def parse_request(text: str) -> dict:
    """'black leggings, size S, under $100' → search_products arguments."""
    args: dict[str, Any] = {}
    if m := _SIZE_RE.search(text):
        args["size"] = m.group(1).upper()
        text = text.replace(m.group(0), " ")
    if m := _PRICE_RE.search(text):
        args["max_price"] = float(m.group(1))
        text = text.replace(m.group(0), " ")
    words = re.findall(r"[a-z']+", text.lower())
    for w in words:
        if w in _COLORS:
            args["color"] = w.capitalize()
        elif w in _CATEGORIES:
            args["category"] = w[:-1] if w.endswith("s") else w  # "leggings" → "legging"
        elif w.replace("'", "") in _DEPARTMENTS:
            args["department"] = _DEPARTMENTS[w.replace("'", "")]
    query = [
        w for w in words
        if w not in _STOPWORDS and w not in _COLORS and w.replace("'", "") not in _DEPARTMENTS
    ]
    args["query"] = " ".join(query)
    return args


class RuleBasedShopper(_FakeBase):
    """search → (recommend, if that tool exists) → short answer."""

    @property
    def _llm_type(self) -> str:
        return "rule-based-fake"

    def _reply(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return AIMessage("", tool_calls=[tool_call("search_products", **self._search_args(messages))])

        if isinstance(last, ToolMessage) and last.name == "search_products":
            products = json.loads(last.content) if last.content.startswith("[") else []
            if not products:
                return AIMessage("I couldn't find anything matching that. Could you loosen the filters?")
            picks = products[:2]
            if "recommend_products" in self.tool_names:
                items = [{"product_id": p["id"], "reason": _reason(p)} for p in picks]
                return AIMessage("", tool_calls=[tool_call("recommend_products", items=items)])
            lines = [f"- {p['title']} (${_price(p)}): {_reason(p)}" for p in picks]
            return AIMessage("Here are my picks:\n" + "\n".join(lines))

        if isinstance(last, ToolMessage) and last.name == "recommend_products":
            return AIMessage("Here are my top picks from the catalog (fake model, no real reasoning).")
        return AIMessage("Done.")

    def _search_args(self, messages) -> dict:
        """Parse the new message; follow-ups ("cheaper ones?", "in black") reuse the last search."""
        text = messages[-1].content
        args = parse_request(text)
        previous = [
            c["args"] for m in messages[:-1] if isinstance(m, AIMessage)
            for c in m.tool_calls if c["name"] == "search_products"
        ]
        if not previous:
            return args
        if re.search(r"\b(cheaper|less expensive)\b", text, re.I):
            return {**previous[-1], "max_price": _cheaper_than(messages, previous[-1])}
        if not args["query"]:  # only filters changed, e.g. "in black" or "size M"
            return {**previous[-1], **{k: v for k, v in args.items() if k != "query"}}
        return args


def _cheaper_than(messages, last_args) -> float:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "search_products" and m.content.startswith("["):
            prices = [_low(p["price"]) for p in json.loads(m.content)[:2]]
            if prices:
                return min(prices) - 0.01
    return last_args.get("max_price", 100) * 0.7


def _low(price) -> float:
    return price[0] if isinstance(price, list) else price


def _price(p: dict) -> str:
    return f"{_low(p['price']):.0f}"


def _reason(p: dict) -> str:
    bits = [p.get("fabric"), p.get("fit"), ", ".join(p.get("activities") or [])]
    detail = " · ".join(b for b in bits if b)
    sale = f", on sale from ${p['was']:.0f}" if p.get("was") else ""
    return f"{detail or p['type']}{sale}"
