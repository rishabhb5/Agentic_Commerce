import json
import os
import time
from pathlib import Path

import httpx
import pytest

from shopper.catalog import Catalog, fetch_raw_products, normalize

FIXTURE = Path(__file__).parent / "fixtures" / "shopify_products.json"
STORE = "https://store.example.com"


def raw_products() -> list[dict]:
    return json.loads(FIXTURE.read_text())["products"]


def paged_transport(products: list[dict], page_size: int = 2, fail: bool = False):
    """Fake Shopify server: serves `products` in pages; records each request."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if fail:
            return httpx.Response(503)
        page = int(request.url.params["page"])
        start = (page - 1) * page_size
        return httpx.Response(200, json={"products": products[start : start + page_size]})

    return httpx.MockTransport(handler), calls


def make_catalog(tmp_path, products=None, fail=False, ttl=3600):
    transport, calls = paged_transport(raw_products() if products is None else products, fail=fail)
    client = httpx.Client(transport=transport)
    catalog = Catalog(store_url=STORE, cache_dir=tmp_path, ttl_seconds=ttl, client=client)
    return catalog, calls


# -- fetching ---------------------------------------------------------------

def test_fetch_reads_every_page_until_empty():
    transport, calls = paged_transport(raw_products(), page_size=2)
    with httpx.Client(transport=transport) as client:
        products = fetch_raw_products(STORE, client=client)
    assert [p["id"] for p in products] == [1001, 1002, 1003, 1004, 1005, 1006]
    # 6 products / 2 per page = 3 full pages, then an empty 4th page ends it.
    assert [c.url.params["page"] for c in calls] == ["1", "2", "3", "4"]


# -- normalizing ------------------------------------------------------------

def test_normalize_basic_fields():
    p = normalize(raw_products()[0], STORE)
    assert p.title == "Airlift High-Waist Legging - Black"
    assert p.url == f"{STORE}/products/airlift-high-waist-legging-black"
    assert (p.department, p.category, p.subcategory) == ("Women", "Bottoms", "Leggings")
    assert p.description == "Our Airlift legging is sleek & sculpting."
    assert (p.fabric, p.fit, p.color_group) == ("Airlift", "Fitted", "Black")
    assert p.activities == ["Pilates", "Yoga"]
    assert p.price_min == p.price_max == 128.0
    assert p.image_url == "https://cdn.example.com/legging-black.jpg"


def test_normalize_only_lists_sizes_in_stock():
    p = normalize(raw_products()[0], STORE)
    assert p.sizes_in_stock == ["XS", "S"]  # M is unavailable
    assert p.in_stock


def test_normalize_handles_size_before_color():
    p = normalize(raw_products()[1], STORE)
    assert p.colors == ["Navy"]
    assert p.sizes_in_stock == ["S", "M"]


def test_normalize_detects_sale_only_when_compare_price_is_higher():
    sale = normalize(raw_products()[1], STORE)
    assert sale.on_sale and sale.compare_at_price == 98.0
    same_price = normalize(raw_products()[2], STORE)  # compare_at == price
    assert not same_price.on_sale and same_price.compare_at_price is None


def test_normalize_skips_internal_and_fake_products():
    assert normalize(raw_products()[4], STORE) is None  # product_type Internal
    assert normalize(raw_products()[5], STORE) is None  # vendor Fake


# -- caching ----------------------------------------------------------------

def test_load_uses_fresh_cache_without_refetching(tmp_path):
    catalog, calls = make_catalog(tmp_path)
    catalog.load()
    assert catalog.loaded_from == "store" and len(catalog) == 4
    fetches = len(calls)

    catalog.load()
    assert catalog.loaded_from == "cache"
    assert len(calls) == fetches  # no new requests


def test_load_refetches_when_cache_expired(tmp_path):
    catalog, calls = make_catalog(tmp_path)
    catalog.load()
    old = time.time() - 7200
    os.utime(catalog.cache_file, (old, old))
    fetches = len(calls)

    catalog.load()
    assert catalog.loaded_from == "store"
    assert len(calls) > fetches


def test_load_falls_back_to_stale_cache_when_store_is_down(tmp_path):
    catalog, _ = make_catalog(tmp_path)
    catalog.load()

    down, _ = make_catalog(tmp_path, fail=True, ttl=0)
    down.load()
    assert down.loaded_from == "stale cache"
    assert len(down) == 4


def test_load_raises_when_store_is_down_and_no_cache(tmp_path):
    catalog, _ = make_catalog(tmp_path, fail=True)
    with pytest.raises(httpx.HTTPError):
        catalog.load()


# -- searching --------------------------------------------------------------

@pytest.fixture
def catalog(tmp_path):
    c, _ = make_catalog(tmp_path)
    return c.load()


def ids(products):
    return [p.id for p in products]


def test_search_matches_plural_and_colour(catalog):
    assert ids(catalog.search("black leggings")) == [1001, 1002]


def test_search_ranks_more_matched_words_first(catalog):
    results = catalog.search("navy leggings")
    assert results[0].id == 1002


def test_search_hides_out_of_stock_by_default(catalog):
    assert 1004 not in ids(catalog.search("bra"))
    assert 1004 in ids(catalog.search("bra", in_stock_only=False))


def test_search_filters(catalog):
    assert ids(catalog.search(size="M")) == [1002, 1003]  # ties: cheapest first
    assert ids(catalog.search("legging", max_price=100)) == [1002]
    assert ids(catalog.search(department="men")) == [1003]
    assert ids(catalog.search(category="leggings", fabric="airlift")) == [1001]
    assert ids(catalog.search(color="blue")) == [1002]  # matches the colour group
    assert ids(catalog.search(activity="yoga")) == [1001]
    assert ids(catalog.search(on_sale=True)) == [1002]


def test_search_no_match(catalog):
    assert catalog.search("tuxedo") == []


def test_get_and_facets(catalog):
    assert catalog.get(1003).title.startswith("Accolade Hoodie")
    assert catalog.get(9999) is None
    facets = catalog.facets()
    assert facets["department"] == [("Women", 3), ("Men", 1)]
    assert ("Airlift", 2) in facets["fabric"]
