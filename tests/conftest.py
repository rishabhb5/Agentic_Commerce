import json
from pathlib import Path

import httpx
import pytest

from shopper.catalog import Catalog
from shopper.tools import set_catalog

FIXTURE = Path(__file__).parent / "fixtures" / "shopify_products.json"


@pytest.fixture
def fixture_catalog(tmp_path):
    """The fixture products, served by a fake Shopify server, wired into the tools."""
    products = json.loads(FIXTURE.read_text())["products"]

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"products": products if page == 1 else []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    catalog = Catalog(store_url="https://store.example.com", cache_dir=tmp_path, client=client).load()
    set_catalog(catalog)
    return catalog
