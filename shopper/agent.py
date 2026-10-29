"""The shopping agent as a LangGraph graph.

    START → agent ──(tool calls?)──▶ tools ──▶ agent ──▶ … ──▶ END

The agent node is the only place the LLM is called. The tools node and the
routing (`tools_condition`) are plain Python. Picks made with
`recommend_products` are copied into `state["recommendations"]`.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Overwrite

from shopper.pricing import DEFAULT_MODEL
from shopper.tools import TOOLS

SHOPPER_PROMPT = """You are a shopping assistant for {store}. You help people find products in this store's catalog and recommend the best ones.

How to work:
- Use search_products to find real products. Never invent products, prices, sizes or colours.
- If the request is vague (no idea of the product type, or size/budget clearly matter), ask one short clarifying question instead of searching.
- Use list_facets when you need to map vague words ("comfy", "hot yoga") to the store's real categories, fabrics and activities.
- If a search finds nothing, retry with fewer or looser filters before giving up.
- Use compare_products when choosing between close options.
- Finish by calling recommend_products with 1-3 picks and a one-line reason each. Then reply briefly: why these, and the main trade-off between them. Product cards with price, sizes and links are shown to the user separately, so don't repeat those details.
- Follow-ups ("cheaper ones?", "in black") refer to the earlier conversation."""


class ShopperState(MessagesState):
    # Validated picks for the current turn. The CLI resets it with Overwrite([])
    # at the start of each user message; within a turn, picks accumulate.
    recommendations: Annotated[list[dict], operator.add]


def make_llm(model: str | None = None) -> BaseChatModel:
    """The chat model. Swap providers by changing this one function."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model or os.getenv("SHOPPER_MODEL", DEFAULT_MODEL),
        max_tokens=4000,
        effort="low",
    )


def new_turn(text: str) -> dict:
    """Graph input for one user message: the message, and cleared picks."""
    return {"messages": [("user", text)], "recommendations": Overwrite([])}


def build_graph(llm: BaseChatModel, store: str = "aloyoga.com", checkpointer=None):
    """The compiled graph. Conversations are saved per `thread_id` by the checkpointer."""
    bind_kwargs = {}
    if getattr(llm, "_llm_type", "") == "anthropic-chat":
        # Prompt caching: cache tools + system prompt + conversation so far.
        # Each later pass re-reads that prefix at ~10% of the input price.
        bind_kwargs["cache_control"] = {"type": "ephemeral"}
    model = llm.bind_tools(TOOLS, **bind_kwargs)
    system = SystemMessage(SHOPPER_PROMPT.format(store=store))
    tool_node = ToolNode(TOOLS)

    def agent(state: ShopperState) -> dict:
        response = model.invoke([system, *state["messages"]])  # ★ the only LLM call
        return {"messages": [response]}

    def tools(state: ShopperState) -> dict:
        out = tool_node.invoke(state)
        picks = [
            pick
            for m in out["messages"]
            if isinstance(m, ToolMessage) and m.name == "recommend_products" and m.artifact
            for pick in m.artifact
        ]
        return {"messages": out["messages"], "recommendations": picks}

    graph = StateGraph(ShopperState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" or END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())
