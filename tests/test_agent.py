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
    assert result["recommendations"][0]["title"] == "Airlift High-Waist Legging - Black"


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


# -- phase 6: recommendations ----------------------------------------------

from shopper.agent import new_turn  # noqa: E402


def test_recommendations_are_validated_and_saved(fixture_catalog):
    llm = ScriptedChatModel(
        responses=[
            AIMessage("", tool_calls=[tool_call("search_products", query="legging")]),
            AIMessage("", tool_calls=[tool_call("recommend_products", items=[
                {"product_id": 1002, "reason": "under budget"},
                {"product_id": 999999, "reason": "made up"},
            ])]),
            AIMessage("The Bootcut is the best value."),
        ]
    )
    result = build_graph(llm).invoke(new_turn("leggings"), CONFIG)

    assert [r["id"] for r in result["recommendations"]] == [1002]
    assert result["recommendations"][0]["reason"] == "under budget"
    feedback = next(m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "recommend_products")
    assert "Rejected unknown ids [999999]" in feedback.content  # the LLM is told
    assert "reason" not in feedback.content  # full product data stays out of the prompt


def test_recommendations_reset_each_turn(fixture_catalog):
    graph = build_graph(RuleBasedShopper())
    first = graph.invoke(new_turn("black leggings"), CONFIG)
    assert first["recommendations"]
    second = graph.invoke(new_turn("mens hoodie"), CONFIG)
    assert [r["title"] for r in second["recommendations"]] == ["Accolade Hoodie - Athletic Grey"]


def test_prompt_caching_only_for_claude(fixture_catalog, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shopper.agent import make_llm

    seen = {}
    llm = make_llm("claude-sonnet-5-5")
    original = type(llm).bind_tools

    def spy(self, tools, **kwargs):
        seen.update(kwargs)
        return original(self, tools, **kwargs)

    monkeypatch.setattr(type(llm), "bind_tools", spy)
    build_graph(llm)
    assert seen["cache_control"] == {"type": "ephemeral"}
