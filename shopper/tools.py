"""LangChain tools the agent can call. Each one is a thin wrapper over the catalog."""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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


@tool
def list_facets() -> str:
    """What the store carries: departments, categories, fabrics, fits, colours and
    activities, with product counts. Use it to map vague requests to real filters."""
    facets = get_catalog().facets()
    return json.dumps({kind: dict(values[:40]) for kind, values in facets.items()})


@tool
def compare_products(product_ids: list[int]) -> str:
    """Side-by-side details (price, fabric, fit, sizes, colours) for 2-5 products."""
    rows = []
    for pid in product_ids[:5]:
        p = get_catalog().get(pid)
        rows.append(
            {**product_summary(p), "description": p.description[:300]}
            if p else {"id": pid, "error": "unknown product id"}
        )
    return json.dumps(rows)


class Recommendation(BaseModel):
    product_id: int = Field(description="id from search_products")
    reason: str = Field(description="one line on why it fits what the user asked for")


@tool(response_format="content_and_artifact")
def recommend_products(items: list[Recommendation]) -> tuple[str, list[dict]]:
    """Submit your final 1-3 picks. Call this before your final answer.
    Only ids returned by the other tools are accepted."""
    accepted, rejected = [], []
    for item in items[:3]:
        item = Recommendation.model_validate(item)
        p = get_catalog().get(item.product_id)
        if p is None:
            rejected.append(item.product_id)
        else:
            accepted.append({**product_summary(p), "reason": item.reason, "image_url": p.image_url})
    message = f"Saved {len(accepted)} pick(s) to show the user."
    if rejected:
        message += f" Rejected unknown ids {rejected}: only recommend products from search results."
    # content → what the LLM sees; artifact → copied into state["recommendations"]
    return message, accepted


TOOLS = [list_facets, search_products, get_product_details, compare_products, recommend_products]
