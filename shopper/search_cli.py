"""Phase 1 demo: search the store's catalog from the terminal (no AI yet).

Examples:
    python -m shopper.search_cli "black leggings" --size S --max-price 100
    python -m shopper.search_cli "hoodie" --department men --on-sale
    python -m shopper.search_cli --facets
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from shopper.catalog import Catalog, Product


def format_product(n: int, p: Product) -> str:
    if p.price_min == p.price_max:
        price = f"${p.price_min:,.2f}"
    else:
        price = f"${p.price_min:,.2f}–${p.price_max:,.2f}"
    if p.on_sale and p.compare_at_price:
        price += f" (was ${p.compare_at_price:,.2f})"
    details = " · ".join(
        x for x in [p.product_type, p.fabric, p.fit, ", ".join(p.activities)] if x
    )
    sizes = ", ".join(p.sizes_in_stock) or "none in stock"
    return (
        f"{n}. {p.title}\n"
        f"   {price}\n"
        f"   {details}\n"
        f"   Colors: {', '.join(p.colors)}   Sizes in stock: {sizes}\n"
        f"   {p.url}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the store catalog.")
    parser.add_argument("query", nargs="?", default="", help='e.g. "black leggings"')
    parser.add_argument("--department", help="Women, Men, Accessories, ...")
    parser.add_argument("--category", help="e.g. Leggings, Bras, Jackets")
    parser.add_argument("--min-price", type=float)
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--color", help="colour name or group, e.g. Black, Navy")
    parser.add_argument("--size", help="e.g. S, M, XL, One Size")
    parser.add_argument("--fabric", help="e.g. Airlift, Airbrush, Alosoft")
    parser.add_argument("--fit", help="Fitted or Flowy")
    parser.add_argument("--activity", help="Yoga, Run, Train, Lounge, ...")
    parser.add_argument("--on-sale", action="store_true", help="only sale items")
    parser.add_argument("--include-out-of-stock", action="store_true")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--facets", action="store_true", help="show what the store carries")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    parser.add_argument("--store-url", help="defaults to $STORE_URL or aloyoga.com")
    args = parser.parse_args(argv)

    catalog = Catalog(store_url=args.store_url)
    started = time.perf_counter()
    try:
        catalog.load(refresh=args.refresh)
    except httpx.HTTPError as exc:
        print(f"Couldn't load products from {catalog.store_url}: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"Loaded {len(catalog):,} products from {catalog.store_url} "
        f"({catalog.loaded_from}, {elapsed:.1f}s)\n"
    )

    if args.facets:
        for kind, values in catalog.facets().items():
            shown = ", ".join(f"{v} ({n})" for v, n in values[:15])
            more = f", … +{len(values) - 15} more" if len(values) > 15 else ""
            print(f"{kind}: {shown}{more}\n")
        return 0

    results = catalog.search(
        args.query,
        department=args.department,
        category=args.category,
        min_price=args.min_price,
        max_price=args.max_price,
        color=args.color,
        size=args.size,
        fabric=args.fabric,
        fit=args.fit,
        activity=args.activity,
        on_sale=True if args.on_sale else None,
        in_stock_only=not args.include_out_of_stock,
        limit=args.limit,
    )
    if not results:
        print("No matching products.")
        return 0
    for n, product in enumerate(results, 1):
        print(format_product(n, product))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
