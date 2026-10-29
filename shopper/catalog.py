"""Product catalog backed by a Shopify store's public products.json feed.

Phase 1 of the shopper: no AI here. This module fetches the store's products,
turns each raw Shopify product into a clean `Product`, caches the result on
disk, and offers keyword search with filters.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path

import httpx

DEFAULT_STORE_URL = "https://www.aloyoga.com"
DEFAULT_CACHE_DIR = Path(".cache")
DEFAULT_TTL_SECONDS = 60 * 60  # refetch the store at most once an hour
PAGE_SIZE = 250  # the maximum Shopify allows per page
MAX_PAGES = 40  # safety cap (Alo has ~19 pages)
USER_AGENT = "Mozilla/5.0 (compatible; agentic-commerce-shopper/0.1)"

# Product types / vendors that aren't real things a shopper can buy.
EXCLUDED_PRODUCT_TYPES = {"internal", "", "gift card", "e gift card"}
EXCLUDED_VENDORS = {"fake"}


@dataclass
class Product:
    id: int
    title: str
    url: str
    department: str  # "Women", "Men", "Accessories", ...
    category: str  # "Bottoms", "Tops", "Bras", ...
    subcategory: str  # "Leggings", "Tanks", ... (may be "")
    product_type: str  # the raw value, e.g. "Women:Bottoms:Leggings"
    description: str
    price_min: float
    price_max: float
    compare_at_price: float | None  # original price when on sale
    on_sale: bool
    colors: list[str]
    color_group: str  # broad colour, e.g. "Black", "Blue"
    sizes_in_stock: list[str]
    in_stock: bool
    fabric: str
    fit: str
    activities: list[str]  # "Yoga", "Run", "Lounge", ...
    image_url: str | None = None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_raw_products(
    store_url: str = DEFAULT_STORE_URL,
    client: httpx.Client | None = None,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """Download every product from `{store_url}/products.json`, page by page.

    Shopify returns at most 250 products per page; an empty page means we've
    reached the end.
    """
    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
    )
    products: list[dict] = []
    try:
        for page in range(1, max_pages + 1):
            response = client.get(
                f"{store_url.rstrip('/')}/products.json",
                params={"limit": PAGE_SIZE, "page": page},
            )
            response.raise_for_status()
            batch = response.json().get("products", [])
            if not batch:
                break
            products.extend(batch)
    finally:
        if own_client:
            client.close()
    return products


# ---------------------------------------------------------------------------
# Normalizing raw Shopify products
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _strip_html(html: str | None) -> str:
    text = _TAG_RE.sub(" ", html or "")
    return _SPACE_RE.sub(" ", unescape(text)).strip()


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    """Values of tags shaped like 'Prefix:Value' (or 'Prefix:Value:...')."""
    out = []
    for tag in tags:
        parts = tag.split(":")
        if len(parts) >= 2 and parts[0] == prefix and parts[1]:
            out.append(parts[1])
    return out


def _option_index(raw: dict, name: str) -> int | None:
    """Position of an option (e.g. "Size") among option1/option2/option3.

    Stores don't keep a fixed order: Alo uses both Color/Size and Size/Color.
    """
    for i, option in enumerate(raw.get("options", [])):
        if option.get("name", "").lower() == name.lower():
            return i
    return None


def _price(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize(raw: dict, store_url: str = DEFAULT_STORE_URL) -> Product | None:
    """Turn one raw Shopify product into a Product, or None to skip it."""
    product_type = (raw.get("product_type") or "").strip()
    vendor = (raw.get("vendor") or "").strip()
    variants = raw.get("variants") or []
    if (
        product_type.lower() in EXCLUDED_PRODUCT_TYPES
        or vendor.lower() in EXCLUDED_VENDORS
        or not variants
    ):
        return None

    type_parts = product_type.split(":")
    department = type_parts[0]
    category = type_parts[1] if len(type_parts) > 1 else ""
    subcategory = type_parts[-1] if len(type_parts) > 2 else ""

    prices = [p for p in (_price(v.get("price")) for v in variants) if p is not None]
    if not prices:
        return None
    compare_prices = [
        c
        for v in variants
        if (c := _price(v.get("compare_at_price"))) is not None
        and c > (_price(v.get("price")) or 0)
    ]

    size_idx = _option_index(raw, "Size")
    color_idx = _option_index(raw, "Color")

    def variant_option(variant: dict, idx: int | None) -> str | None:
        return variant.get(f"option{idx + 1}") if idx is not None else None

    sizes_in_stock: list[str] = []
    colors: list[str] = []
    for v in variants:
        color = variant_option(v, color_idx)
        if color and color not in colors:
            colors.append(color)
        size = variant_option(v, size_idx)
        if v.get("available") and size and size not in sizes_in_stock:
            sizes_in_stock.append(size)

    tags = raw.get("tags") or []
    if isinstance(tags, str):  # some Shopify feeds send a comma-separated string
        tags = [t.strip() for t in tags.split(",")]
    activities = sorted(
        {t.split(":")[2] for t in tags if t.count(":") >= 2 and t.split(":")[1] == "Activity"}
    )
    fabric = next(iter(_tag_values(tags, "Fabric")), "")
    fit = next(iter(_tag_values(tags, "Fit")), "")
    color_group = next(iter(_tag_values(tags, "ColorGroup")), "")
    images = raw.get("images") or []

    title = raw.get("title", "").strip()
    description = _strip_html(raw.get("body_html"))

    return Product(
        id=raw["id"],
        title=title,
        url=f"{store_url.rstrip('/')}/products/{raw.get('handle', '')}",
        department=department,
        category=category,
        subcategory=subcategory,
        product_type=product_type,
        description=description,
        price_min=min(prices),
        price_max=max(prices),
        compare_at_price=max(compare_prices) if compare_prices else None,
        on_sale=bool(compare_prices),
        colors=colors,
        color_group=color_group,
        sizes_in_stock=sizes_in_stock,
        in_stock=any(v.get("available") for v in variants),
        fabric=fabric,
        fit=fit,
        activities=activities,
        image_url=images[0].get("src") if images else None,
    )


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Very small plural stripper so 'leggings' matches 'legging'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower())]


def _matches(value: str, wanted: str | None) -> bool:
    return wanted is None or wanted.lower() in value.lower()


# Where a query word is found decides how much it counts.
_FIELD_WEIGHTS = (
    ("title", 3.0),
    ("type", 3.0),
    ("attributes", 2.0),
    ("description", 1.0),
)


def _field_words(p: Product) -> dict[str, set[str]]:
    return {
        "title": set(_words(p.title)),
        "type": set(_words(f"{p.department} {p.category} {p.subcategory}")),
        "attributes": set(
            _words(" ".join([p.fabric, p.fit, p.color_group, *p.colors, *p.activities]))
        ),
        "description": set(_words(p.description)),
    }


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class Catalog:
    """Loads, caches and searches a store's products."""

    def __init__(
        self,
        store_url: str | None = None,
        cache_dir: Path | str | None = None,
        ttl_seconds: int | None = None,
        client: httpx.Client | None = None,
    ):
        self.store_url = (store_url or os.getenv("STORE_URL") or DEFAULT_STORE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else int(os.getenv("CATALOG_TTL_SECONDS", DEFAULT_TTL_SECONDS))
        )
        self._client = client
        self._products: dict[int, Product] = {}
        self._field_index: dict[int, dict[str, set[str]]] = {}
        self.loaded_from: str = ""  # "cache" or "store", for display

    # -- loading -----------------------------------------------------------

    @property
    def cache_file(self) -> Path:
        host = re.sub(r"[^a-z0-9.]+", "_", self.store_url.lower().split("://")[-1])
        return self.cache_dir / f"{host}.products.json"

    def _cache_is_fresh(self) -> bool:
        return (
            self.cache_file.exists()
            and time.time() - self.cache_file.stat().st_mtime < self.ttl_seconds
        )

    def load(self, refresh: bool = False) -> "Catalog":
        """Load products from the disk cache if it's fresh, else from the store.

        If the store can't be reached but an old cache exists, use the old
        cache rather than failing.
        """
        raw: list[dict] | None = None
        if not refresh and self._cache_is_fresh():
            raw = json.loads(self.cache_file.read_text())
            self.loaded_from = "cache"
        else:
            try:
                raw = fetch_raw_products(self.store_url, client=self._client)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(json.dumps(raw))
                self.loaded_from = "store"
            except httpx.HTTPError:
                if not self.cache_file.exists():
                    raise
                raw = json.loads(self.cache_file.read_text())
                self.loaded_from = "stale cache"

        self._products = {}
        for item in raw:
            product = normalize(item, self.store_url)
            if product is not None:
                self._products[product.id] = product
        self._field_index = {pid: _field_words(p) for pid, p in self._products.items()}
        return self

    def _ensure_loaded(self) -> None:
        if not self._products:
            self.load()

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._products)

    def all(self) -> list[Product]:
        self._ensure_loaded()
        return list(self._products.values())

    def get(self, product_id: int) -> Product | None:
        self._ensure_loaded()
        return self._products.get(int(product_id))

    def search(
        self,
        query: str = "",
        *,
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
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[Product]:
        """Keyword search plus filters.

        Filters narrow the list first. Then each product is scored by where
        the query words appear (title and type count most, description least).
        Products matching more distinct query words always rank higher.
        """
        self._ensure_loaded()
        query_words = list(dict.fromkeys(_words(query)))
        results: list[tuple[int, float, float, Product]] = []

        for p in self._products.values():
            if department and p.department.lower() != department.lower():
                continue
            if category and category.lower() not in p.product_type.lower():
                continue
            if min_price is not None and p.price_max < min_price:
                continue
            if max_price is not None and p.price_min > max_price:
                continue
            if color and not any(
                _matches(c, color) for c in [*p.colors, p.color_group]
            ):
                continue
            if size and size.lower() not in (s.lower() for s in p.sizes_in_stock):
                continue
            if not _matches(p.fabric, fabric) or not _matches(p.fit, fit):
                continue
            if activity and not any(_matches(a, activity) for a in p.activities):
                continue
            if on_sale is not None and p.on_sale != on_sale:
                continue
            if in_stock_only and not p.in_stock:
                continue

            matched, score = 0, 0.0
            fields = self._field_index[p.id]
            for word in query_words:
                best = max(
                    (w for name, w in _FIELD_WEIGHTS if word in fields[name]),
                    default=0.0,
                )
                if best:
                    matched += 1
                    score += best
            if query_words and matched == 0:
                continue
            results.append((matched, score, -p.price_min, p))

        results.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        return [p for *_, p in results[:limit]]

    def facets(self) -> dict[str, list[tuple[str, int]]]:
        """What the store carries, with counts, most common first."""
        self._ensure_loaded()
        counts: dict[str, dict[str, int]] = {
            k: {} for k in ("department", "category", "fabric", "fit", "color_group", "activity")
        }

        def bump(kind: str, value: str) -> None:
            if value:
                counts[kind][value] = counts[kind].get(value, 0) + 1

        for p in self._products.values():
            bump("department", p.department)
            bump("category", p.product_type)
            bump("fabric", p.fabric)
            bump("fit", p.fit)
            bump("color_group", p.color_group)
            for a in p.activities:
                bump("activity", a)
        return {
            kind: sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
            for kind, values in counts.items()
        }


def product_to_dict(p: Product) -> dict:
    """Plain dict, handy for printing or JSON."""
    return asdict(p)


def product_summary(p: Product) -> dict:
    """The fields an LLM needs to judge a product, without the long description."""
    return {
        "id": p.id,
        "title": p.title,
        "type": p.product_type,
        "price": p.price_min if p.price_min == p.price_max else [p.price_min, p.price_max],
        **({"was": p.compare_at_price} if p.on_sale else {}),
        "fabric": p.fabric,
        "fit": p.fit,
        "activities": p.activities,
        "colors": p.colors[:8],
        "sizes_in_stock": p.sizes_in_stock,
        "url": p.url,
    }
