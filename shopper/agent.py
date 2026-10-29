"""The shopping agent as a LangGraph graph.

    START → agent ──(tool calls?)──▶ tools ──▶ agent ──▶ … ──▶ END

The agent node is the only place the LLM is called. The tools node and the
routing (`tools_condition`) are plain Python.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from shopper.pricing import DEFAULT_MODEL
from shopper.tools import TOOLS

SHOPPER_PROMPT = """You are a shopping assistant for {store}.
- Use search_products to find real products. Never invent products, prices or sizes.
- If a search finds nothing, try again with fewer or looser filters.
- Recommend 1-3 products: name, price, and one line on why each fits.
- Keep replies short."""


def make_llm(model: str | None = None) -> BaseChatModel:
    """The chat model. Swap providers by changing this one function."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model or os.getenv("SHOPPER_MODEL", DEFAULT_MODEL),
        max_tokens=4000,
        effort="low",
    )


def build_graph(llm: BaseChatModel, store: str = "aloyoga.com"):
    model = llm.bind_tools(TOOLS)
    system = SystemMessage(SHOPPER_PROMPT.format(store=store))

    def agent(state: MessagesState) -> dict:
        response = model.invoke([system, *state["messages"]])  # ★ the only LLM call
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" or END
    graph.add_edge("tools", "agent")
    return graph.compile()
