# Agentic_Commerce

An AI shopping assistant built in phases (see [docs/PLAN.md](docs/PLAN.md)).
It shops the live [Alo Yoga](https://www.aloyoga.com) catalog through the
store's public Shopify `products.json` feed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

Phase 1 needs no key. Phases 2–6 call Claude (`claude-sonnet-5-5` by default;
set `SHOPPER_MODEL` to change it). Everything can also run offline with
`--fake-llm`, a rule-based stand-in model.

## Run the shopper

```bash
python -m shopper                  # chat: You: prompt, /new, /quit (or Ctrl-D)
python -m shopper --verbose        # also show each agent/tools pass, tokens and cost
python -m shopper "black leggings, size S, under $100"   # one question, then exit
python -m shopper --fake-llm       # offline, no API key
```

Follow-ups like "cheaper ones?" or "in navy" work because the conversation
is saved per thread. Recommendations are printed as product cards.

## Phase 1: catalog search (no AI)

`shopper/catalog.py` downloads every product (~19 pages, ~13s), cleans each
one up, and caches the result in `.cache/` for an hour so later runs take
about a second.

```bash
# keyword search + filters
python -m shopper.search_cli "black leggings" --size S --max-price 100
python -m shopper.search_cli "hoodie" --department men --on-sale

# what the store carries (departments, categories, fabrics, colours, activities)
python -m shopper.search_cli --facets

# all options
python -m shopper.search_cli --help
```

Use `--refresh` to ignore the cache, or set `STORE_URL` to try another
Shopify store.

## Phase 2: one LLM call, no tools

```bash
python -m learn.phase2_one_call "black leggings, size S, under $100"
```

Prints Claude's reply, the tokens and the cost. Claude can't see the catalog,
so the products it names are guesses.

## Phase 3: tools by hand

```bash
python -m learn.phase3_tools_by_hand "black leggings, size S, under $100"
```

Gives Claude a `search_products` tool and runs the loop by hand: print the
tool call Claude asks for, run it, send the result back, repeat until it
answers.

## Phases 4–6: LangGraph runs the loop

`shopper/agent.py` builds the graph: an `agent` node (the only LLM call),
a `tools` node, and `tools_condition` routing between them.

```
START → agent ──(tool calls?)──▶ tools ──▶ agent ──▶ … ──▶ END
```

- Phase 4: the graph replaces phase 3's hand-written loop.
- Phase 5: `MemorySaver` + `thread_id` keep the conversation, so follow-ups work.
- Phase 6: `list_facets`, `compare_products` and `recommend_products` (picks
  are checked against the catalog and saved to `state["recommendations"]`),
  product cards, and prompt caching.

## Tests

```bash
pytest
```

Tests use a small sample in `tests/fixtures/`, a fake HTTP server and a
scripted fake chat model, so they run offline with no API key.
