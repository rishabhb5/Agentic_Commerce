# Plan: Agentic Commerce Shopper (LangGraph, terminal first)

## Context
The repo (`rishabhb5/Agentic_Commerce`) holds only `README.md` and a hello-world `index.html`. The user wants an AI shopping assistant. Agreed scope:
- **Find & recommend only**: the user describes what they want, the agent searches the catalog, compares options and recommends items with reasons. No cart/checkout.
- **Live catalog** from a Shopify store's public `products.json` (default `https://www.aloyoga.com`, configurable via `STORE_URL`). Verified reachable: 250 products/page with title, `product_type` like `Women:Tops:Tanks`, tags (`Fabric:*`, `Fit:*`, `ColorGroup:*`, `<Dept>:Activity:*`, many internal ones), variants with price, `compare_at_price`, per-size `available`, Color/Size options.
- **Agent loop in LangGraph**, used from a **terminal chat** for now. FastAPI + React come later as a layer over the same graph; nothing here depends on them.

Model: default Claude via `langchain-anthropic` (`claude-sonnet-5-5`), configurable with `ANTHROPIC_API_KEY` / `SHOPPER_MODEL`; swapping providers means changing one function in `agent.py`.

Existing `index.html` is left untouched.

## Working rules
- **No commits without explicit approval.** After each change, show what changed and wait. Commit, push, open or merge a PR only after the user replies **"Y"** or **"yes"**. Approval covers that one change only; ask again for the next one.

## Build phases
Each phase adds one idea, ends with something runnable, and is its own PR (subject to the working rules above). Phases 2–3 are learning scripts kept in `learn/` so they can be compared with the LangGraph version.

1. **Catalog, no AI**: fetch + normalize Alo `products.json`, `search()`; a script prints results for e.g. `"black leggings" size=S max=100`. Shows the real data and filters.
2. **One LLM call, no tools** (`learn/`): send a message to Claude, print reply + tokens + cost. Shows Claude guessing products because it has no catalog access.
3. **Tools by hand** (`learn/`): wrap `search()` as a tool; print the tool call Claude requests, run it, send the result back, print the answer. One loop pass done manually.
4. **LangGraph runs the loop**: agent node + tools node + `tools_condition`; `--verbose` prints each pass. Compare with phase 3 to see what LangGraph automates.
5. **Memory + chat**: `MemorySaver` + `thread_id`, `You:` loop with `/new` and `/quit`. Follow-ups like "cheaper ones" work.
6. **Good shopper**: compare / facets / validated `recommend_products`, product panels, prompt caching, fake-model tests. Real-products-only picks and lower cost per message.

Phase 1 needs no API key; phases 2–6 need `ANTHROPIC_API_KEY` for real replies (tests use a fake model).

## Layout
```
shopper/
  __init__.py
  catalog.py      fetch + cache + normalize + search (pure Python, no LLM)
  tools.py        LangChain @tool wrappers
  agent.py        LangGraph graph + system prompt + build_graph(llm)
  cli.py          terminal chat loop (python -m shopper)
  __main__.py
tests/
  fixtures/shopify_products.json   small sample in Shopify's format
  test_catalog.py
  test_agent.py
requirements.txt  langgraph, langchain-anthropic, httpx, python-dotenv, rich, pytest
.env.example      ANTHROPIC_API_KEY=, SHOPPER_MODEL=claude-sonnet-5-5, STORE_URL=https://www.aloyoga.com
README.md         setup + run instructions
```

## Catalog (`catalog.py`)
- Fetch `{STORE_URL}/products.json?limit=250&page=N` with `httpx`, paging until an empty page (cap 40 pages), browser-like User-Agent, timeout.
- In-memory cache with TTL (default 1h, `CATALOG_TTL_SECONDS`); keep the stale copy if a refresh fails.
- Normalize to: `id, title, department, category, subcategory` (split `product_type`), `fabric, fit, color_group, activities` (from tags), `description` (HTML stripped), `price_min/price_max`, `compare_at_price` + `on_sale`, `colors`, `sizes_in_stock`, `in_stock`, `image_url`, `url = {STORE_URL}/products/{handle}`. Internal tags are dropped.
- `search(query, department?, category?, min_price?, max_price?, color?, size?, fabric?, fit?, activity?, on_sale?, in_stock_only=True, limit=8)`: keyword scoring over title/type/useful tags/description, then filters.
- `get(product_id)`, `facets()` → available departments/categories/fabrics/fits/activities.

## Tools (`tools.py`)
- `search_products(...)` (filters above) → compact matches.
- `list_facets()` → what the store carries, so the agent can map vague requests to real categories.
- `get_product_details(product_id)`, `compare_products(product_ids[])`.
- `recommend_products(items: [{product_id, reason}])` → final picks, validated against the catalog (unknown ids rejected).

## Graph (`agent.py`)
- `StateGraph` over `MessagesState` + `recommendations`; nodes `agent` (model `.bind_tools`) and `tools` (`ToolNode`), `tools_condition` loop, END when no tool calls.
- `recommend_products` results copied into `state["recommendations"]`, reset each user turn.
- `MemorySaver` checkpointer keyed by `thread_id` for follow-ups ("cheaper", "in black").
- `build_graph(llm)` takes the model so tests can inject a fake.
- System prompt: only recommend catalog items; ask a clarifying question when too vague (size, budget, use); 1–3 picks via `recommend_products`; brief trade-offs.

## How the agent loop works

```
                 You: "black leggings, size S, under $100"
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │            START             │
                    │  add your message to the     │
                    │  conversation (per thread)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │                   agent node                      │
   ┌────▶│  ★ LLM CALL (one per pass)                        │
   │      │  Claude reads: system prompt + full conversation  │
   │      │  + results of any tools it already called         │
   │      │  and decides: call tool(s)?  or  answer?          │
   │      └───────────────────────┬──────────────────────────┘
   │                              │
   │                     tools_condition
   │                 (did it request a tool?)
   │                   │                     │
   │                  yes                    no
   │                   ▼                     ▼
   │      ┌─────────────────────────┐   ┌───────────────────────┐
   │      │       tools node        │   │          END          │
   │      │  runs the requested     │   │  reply + saved        │
   │      │  tool(s) on the catalog │   │  recommendations go   │
   │      │                         │   │  back to the terminal │
   │      │  • list_facets          │   └───────────────────────┘
   │      │  • search_products      │
   │      │  • get_product_details  │
   │      │  • compare_products     │
   │      │  • recommend_products ──┼──▶ saved to state.recommendations
   │      └────────────┬────────────┘     (unknown product IDs rejected)
   │                   │ tool results added to the conversation
   └───────────────────┘

   Catalog (behind the tools): aloyoga.com/products.json → fetched,
   cleaned up, kept in memory for 1 hour
   Memory (MemorySaver): conversation saved per thread_id, so
   "cheaper ones?" next turn still knows the context
```

Example run for one request:

| Pass | LLM decides | Tool result fed back |
|---|---|---|
| 1 | `search_products(query="leggings", color="black", size="S", max_price=100)` | 8 matches |
| 2 | `compare_products([3 best ids])` | side-by-side fabric, fit, price |
| 3 | `recommend_products([{id, reason}, …])` | picks checked against the catalog and saved |
| 4 | no tool call, writes the answer | → END, terminal prints reply + product boxes |

### Where the LLM call happens
Only in the agent node; the tools node and `tools_condition` are plain Python.

```python
llm = ChatAnthropic(model="claude-sonnet-5-5").bind_tools(TOOLS)

def agent_node(state):
    messages = [SystemMessage(SHOPPER_PROMPT), *state["messages"]]
    response = llm.invoke(messages)        # ★ the only LLM call
    return {"messages": [response]}         # may contain tool_calls
```

- **In:** system prompt + whole conversation + all earlier tool results (+ tool definitions via `bind_tools`).
- **Out:** either `tool_calls` (route to tools) or plain text (route to END).
- **Cost:** one call per pass, each resending the growing conversation; the example above is 4 calls for one user message.

### What the LLM call does
1. **Turns words into a structured action**: "something comfy for hot yoga, not too pricey" → `search_products(query="hot yoga", activity="Yoga", fabric="Airlift", max_price=90)`.
2. **Reads results and decides the next step**: compare, search again with looser filters, ask a clarifying question, or recommend and explain trade-offs.

| LLM call (thinking) | Python tools (doing) |
|---|---|
| understands your request | fetch the Alo catalog |
| picks tools and filter values | filter by price, size, colour |
| judges which results fit you | look up product details |
| explains trade-offs in plain English | check recommended IDs are real |

## Cost (estimates, Claude Sonnet 5.5: $2 / $10 per 1M input / output tokens, $0.20 cached input)
- One LLM call: ~2k–5k input tokens + ~100–600 output (incl. thinking) ≈ **~1¢**.
- One user message (≈4 calls, see example above): ~15k input + ~2k output ≈ **~5¢**; grows with conversation length (~20¢/message by turn 10).
- **Prompt caching** on the tools + system prompt + history prefix cuts input cost ~10× on repeat reads → ~2–3¢/message. Include it in the build.
- `--verbose` prints tokens and cost per reply so real numbers replace these estimates.
- Alternatives: Haiku 5.5 ≈ 0.3¢/message, Opus 5.5 ≈ 10¢/message.

## Terminal chat (`cli.py`)
- `python -m shopper` → loads `.env`, warms the catalog ("Loaded 1,234 products from aloyoga.com"), then a `You:` prompt loop.
- Prints the agent reply, then each recommendation as a compact `rich` panel: title, price (sale price struck through), colors, in-stock sizes, reason, product URL.
- `--verbose` shows tool calls as they happen; `/new` starts a fresh thread; `/quit` or Ctrl-D exits.
- Clear messages if `ANTHROPIC_API_KEY` is missing or the store can't be reached.

## Verification
1. `pip install -r requirements.txt && pytest`: catalog tests on the fixture with HTTP mocked (normalization, paging, cache, filters); agent test drives the full graph with a scripted fake chat model (`search_products` → `recommend_products`), asserting validated recommendations and that a second turn on the same thread keeps history.
2. Live-store check: load the real Alo feed and run a few `search()` calls (e.g. black leggings size S under $100), printing results.
3. Run `python -m shopper` with piped input. With a real `ANTHROPIC_API_KEY` this is end-to-end; without one I'll run it with the fake model behind a `--fake-llm` flag and say so.
4. Restart `claude/adoring-johnson-n1m9z8` from current `main`, commit, push, open a PR.
