# Agentic_Commerce

An AI shopping assistant built in phases (see [docs/PLAN.md](docs/PLAN.md)).
It shops the live [Alo Yoga](https://www.aloyoga.com) catalog through the
store's public Shopify `products.json` feed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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

## Tests

```bash
pytest
```

Tests use a small sample in `tests/fixtures/` and a fake HTTP server, so
they run offline.
