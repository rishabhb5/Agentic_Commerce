import json

from langchain_core.messages import AIMessage, ToolMessage

from shopper.agent import build_graph
from shopper.fake_llm import RuleBasedShopper, ScriptedChatModel, parse_request, tool_call


def test_graph_runs_tool_then_answers(fixture_catalog):
    llm = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[tool_call("search_products", query="leggings", max_price=100)]),
            AIMessage("Try the Airbrush Bootcut Legging."),
        ]
    )
    result = build_graph(llm).invoke({"messages": [("user", "leggings under $100")]})

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
        "size": "S", "max_price": 100.0, "color": "Black", "query": "leggings",
    }


def test_rule_based_shopper_end_to_end(fixture_catalog):
    result = build_graph(RuleBasedShopper()).invoke({"messages": [("user", "black leggings size S")]})
    assert "Airlift High-Waist Legging - Black" in result["messages"][-1].content
