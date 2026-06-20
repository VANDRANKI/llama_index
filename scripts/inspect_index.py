#!/usr/bin/env python3
"""Inspect a persisted LlamaIndex vector index and print node statistics.

Useful for auditing index contents without running a full query.

Usage::

    python scripts/inspect_index.py --persist-dir ./storage
    python scripts/inspect_index.py --persist-dir ./storage --sample 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the index inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect a persisted LlamaIndex vector index."
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        type=Path,
        help="Directory containing the persisted index (created by storage_context.persist()).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Print this many sample node contents (default: 0 = none).",
    )
    args = parser.parse_args()

    if not args.persist_dir.exists():
        print(
            f"ERROR: Persist directory '{args.persist_dir}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except ImportError:
        print(
            "ERROR: llama_index is not installed. Run: pip install llama-index",
            file=sys.stderr,
        )
        sys.exit(1)

    storage_context = StorageContext.from_defaults(persist_dir=str(args.persist_dir))
    index = load_index_from_storage(storage_context)

    docstore = storage_context.docstore
    nodes = list(docstore.docs.values())
    print(f"Index loaded from: {args.persist_dir}")
    print(f"Total nodes: {len(nodes)}")

    if nodes:
        text_lengths = [len(n.get_content()) for n in nodes]
        print(f"Avg node length (chars): {sum(text_lengths) / len(text_lengths):.0f}")
        print(f"Min node length (chars): {min(text_lengths)}")
        print(f"Max node length (chars): {max(text_lengths)}")

    if args.sample > 0:
        print(f"\nSample nodes (first {args.sample}):")
        for node in nodes[: args.sample]:
            content = node.get_content()
            preview = content[:200].replace("\n", " ")
            print(f"  [{node.node_id[:8]}] {preview}...")


if __name__ == "__main__":
    main()
