"""LangChain tools the agent can call. Each one is a thin wrapper over the catalog."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from shopper.catalog import Catalog, product_summary

_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog().load()
    return _catalog


def set_catalog(catalog: Catalog) -> None:
    """Point the tools at a specific catalog (tests use a fixture catalog)."""
    global _catalog
    _catalog = catalog


@tool
def search_products(
    query: str,
    department: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    color: str | None = None,
    size: str | None = None,
    fabric: str | None = None,
    fit: str | None = None,
    activity: str | None = None,
    on_sale: bool | None = None,
    limit: int = 8,
) -> str:
    """Search the store catalog. Only in-stock products are returned.

    Args:
        query: keywords, e.g. "leggings" or "zip hoodie"
        department: Women, Men, Accessories, ...
        category: e.g. Leggings, Bras, Jackets, Shorts
        min_price: lowest price in USD
        max_price: highest price in USD
        color: colour name or group, e.g. Black, Navy, Blue
        size: a size that must be in stock, e.g. XS, S, M, L, 30
        fabric: e.g. Airlift, Airbrush, Alosoft
        fit: Fitted or Flowy
        activity: Yoga, Run, Train, Lounge, Tennis, ...
        on_sale: true for sale items only
        limit: max results (default 8)
    """
    results = get_catalog().search(
        query,
        department=department,
        category=category,
        min_price=min_price,
        max_price=max_price,
        color=color,
        size=size,
        fabric=fabric,
        fit=fit,
        activity=activity,
        on_sale=on_sale,
        limit=min(limit, 20),
    )
    return json.dumps([product_summary(p) for p in results])


@tool
def get_product_details(product_id: int) -> str:
    """Full details for one product, including its description."""
    p = get_catalog().get(product_id)
    if p is None:
        return json.dumps({"error": f"No product with id {product_id}"})
    return json.dumps({**product_summary(p), "description": p.description[:1200]})


TOOLS = [search_products, get_product_details]
