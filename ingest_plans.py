"""Ingest the plan & pricing catalog into the ``plans`` collection.

Each plan / add-on in data/plans.json becomes one vector document, mirroring the
pattern of the other ingest scripts (1 item = 1 document). The JSON has a mixed
schema — subscription plans price via ``monthly_price`` while add-ons price via
``price`` + ``price_unit`` and add a ``coverage`` field — so each item is
rendered into a readable, fully-labelled text block to keep pricing accurate in
both the embedding and the LLM context. Re-running is idempotent (FR-17).

Support ops can update data/plans.json and re-run this without a code release.

    python ingest_plans.py
"""

from __future__ import annotations

import json

from langchain_core.documents import Document

from config import COLLECTIONS, DATA_DIR
from vectorstore import reset_collection

PLANS_JSON = DATA_DIR / "plans.json"


def _format_data(plan: dict) -> str | None:
    """Describe the data allowance, handling unlimited / capped / textual values."""
    if plan.get("data_unlimited"):
        return "Unlimited data"
    data = plan.get("data_gb")
    if data is None:
        return None
    # data_gb is usually a number but can be descriptive text (e.g. day passes).
    return f"{data} GB data" if isinstance(data, (int, float)) else f"Data: {data}"


def _format_price(plan: dict, currency: str) -> str | None:
    """Describe the price for both subscription plans and add-ons."""
    if plan.get("monthly_price") is not None:
        return f"Price: {currency} ${plan['monthly_price']:.2f} per month"
    if plan.get("price") is not None:
        unit = plan.get("price_unit", "")
        unit = f" {unit}" if unit else ""
        return f"Price: {currency} ${plan['price']:.2f}{unit}"
    return None


def _plan_to_text(plan: dict, currency: str) -> str:
    """Render a single plan/add-on into a labelled, human-readable block."""
    kind = "Add-on" if plan.get("type") == "add-on" else "Plan"
    lines = [f"{kind}: {plan['name']} ({plan.get('type', 'plan')})"]

    for value in (
        _format_price(plan, currency),
        _format_data(plan),
    ):
        if value:
            lines.append(value)

    # Simple scalar fields rendered as "Label: value" when present.
    scalar_fields = [
        ("talk_minutes", "Talk"),
        ("texts", "Texts"),
        ("hotspot_gb", "Hotspot (GB)"),
        ("network", "Network"),
        ("coverage", "Coverage"),
        ("contract", "Contract"),
        ("intro_offer", "Intro offer"),
        ("best_for", "Best for"),
    ]
    for key, label in scalar_fields:
        value = plan.get(key)
        if value is not None and value != "":
            lines.append(f"{label}: {value}")

    features = plan.get("features")
    if features:
        lines.append("Features: " + "; ".join(features))

    return "\n".join(lines)


def load_documents() -> list[Document]:
    with open(PLANS_JSON, encoding="utf-8") as f:
        catalog = json.load(f)

    currency = catalog.get("currency", "USD")
    docs: list[Document] = []
    for plan in catalog.get("plans", []):
        docs.append(
            Document(
                page_content=_plan_to_text(plan, currency),
                metadata={
                    "source": "PLANS",
                    "id": plan["id"],
                    "name": plan["name"],
                    "type": plan.get("type", ""),
                    "category": plan.get("category", ""),
                },
            )
        )
    return docs


def main() -> None:
    docs = load_documents()
    store = reset_collection(COLLECTIONS["PLANS"])
    store.add_documents(docs, ids=[f"plan-{d.metadata['id']}" for d in docs])
    print(f"Ingested {len(docs)} plans into '{COLLECTIONS['PLANS']}'.")


if __name__ == "__main__":
    main()
