import json

from langchain_core.messages import AIMessage, ToolMessage

from shopper.agent import build_graph
from shopper.fake_llm import RuleBasedShopper, ScriptedChatModel, parse_request, tool_call

CONFIG = {"configurable": {"thread_id": "test"}}


def test_graph_runs_tool_then_answers(fixture_catalog):
    llm = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[tool_call("search_products", query="leggings", max_price=100)]),
            AIMessage("Try the Airbrush Bootcut Legging."),
        ]
    )
    result = build_graph(llm).invoke({"messages": [("user", "leggings under $100")]}, CONFIG)

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    found = json.loads(tool_msgs[0].content)
    assert [p["title"] for p in found] == ["Airbrush Bootcut Legging - Navy"]
    assert result["messages"][-1].content == "Try the Airbrush Bootcut Legging."
    # Second LLM call saw the system prompt, the question, its tool call and the result.
    second_call = llm.calls[1]
    assert second_call[0].type == "system"
    assert isinstance(second_call[-1], ToolMessage)


def test_parse_request():
    assert parse_request("black leggings, size S, under $100") == {
        "size": "S", "max_price": 100.0, "color": "Black", "category": "legging", "query": "leggings",
    }


def test_rule_based_shopper_end_to_end(fixture_catalog):
    result = build_graph(RuleBasedShopper()).invoke({"messages": [("user", "black leggings size S")]}, CONFIG)
    assert "Airlift High-Waist Legging - Black" in result["messages"][-1].content


def test_second_turn_on_same_thread_keeps_history(fixture_catalog):
    graph = build_graph(RuleBasedShopper())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [("user", "leggings")]}, config)
    result = graph.invoke({"messages": [("user", "cheaper ones?")]}, config)

    searches = [
        c["args"] for m in result["messages"] if isinstance(m, AIMessage)
        for c in m.tool_calls if c["name"] == "search_products"
    ]
    assert len(searches) == 2  # both turns are in the saved conversation
    assert searches[1]["query"] == "leggings"  # the follow-up reused the first search
    assert searches[1]["max_price"] < 128

    other = graph.invoke({"messages": [("user", "cheaper ones?")]}, {"configurable": {"thread_id": "t2"}})
    assert len(other["messages"]) < len(result["messages"])  # a new thread starts empty
