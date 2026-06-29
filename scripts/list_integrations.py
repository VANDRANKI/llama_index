#!/usr/bin/env python3
"""List installed LlamaIndex integration packages grouped by category.

Scans importlib.metadata for packages whose name starts with
'llama-index-' and groups them by the second name component
(llms, embeddings, vector-stores, readers, tools, etc.).

Usage:
    python scripts/list_integrations.py
    python scripts/list_integrations.py --category llms
    python scripts/list_integrations.py --json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections import defaultdict

PACKAGE_PREFIX = "llama-index-"


def get_installed_integrations() -> dict[str, list[dict]]:
    """Return installed llama-index-* packages grouped by category."""
    by_category: dict[str, list[dict]] = defaultdict(list)

    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] or ""
        if not name.lower().startswith(PACKAGE_PREFIX):
            continue
        # llama-index-<category>-<provider> -> category = second component
        parts = name[len(PACKAGE_PREFIX):].split("-", 1)
        category = parts[0] if parts else "other"
        version = dist.metadata["Version"] or "unknown"
        by_category[category].append({"name": name, "version": version})

    # Sort within each category by name
    return {cat: sorted(pkgs, key=lambda p: p["name"]) for cat, pkgs in sorted(by_category.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        help="Only show packages in this category (e.g. llms, embeddings)",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args(argv)

    integrations = get_installed_integrations()

    if args.category:
        integrations = {k: v for k, v in integrations.items() if k == args.category}
        if not integrations:
            print(f"No installed integrations found for category: {args.category}")
            return 1

    if not integrations:
        print("No llama-index-* integration packages installed.")
        return 0

    if args.as_json:
        print(json.dumps(integrations, indent=2))
        return 0

    total = sum(len(v) for v in integrations.values())
    print(f"Installed LlamaIndex integrations ({total} total):\n")
    for category, packages in integrations.items():
        print(f"  {category} ({len(packages)}):")
        for pkg in packages:
            print(f"    {pkg['name']:50s} {pkg['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
