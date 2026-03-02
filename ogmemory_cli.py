#!/usr/bin/env python3
"""oG-Memory CLI tool for Claude Code plugin"""

import asyncio
import argparse
import sys
from pathlib import Path

# 需先执行 uv sync 或 pip install -e . 以安装 ogmemory 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ogmemory import MemoryEngine


async def cmd_search(args):
    """Search memories"""
    engine = MemoryEngine()
    results = await engine.search(
        args.query,
        limit=args.top_k,
        use_hybrid=args.hybrid,
    )

    if args.json_output:
        import json

        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "text": r.text,
                        "score": r.score,
                        "source": r.source,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.score:.4f}] {r.text[:100]}...")
            print(f"   Source: {r.source}")
            if r.id:
                print(f"   Chunk ID: {r.id}  (use: python3 -m ogmemory expand {r.id})")
            print()

    engine.close()


async def cmd_index(args):
    """Index markdown files"""
    engine = MemoryEngine()

    path = Path(args.path)
    if path.is_file():
        count = await engine.index_file(str(path))
    elif path.is_dir():
        count = await engine.index_directory(str(path))
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return

    print(f"Indexed {count} chunks")
    engine.close()


async def cmd_stats(args):
    """Show statistics"""
    engine = MemoryEngine()
    stats = engine.get_stats()

    print("oG-Memory Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")

    engine.close()


async def cmd_reset(args):
    """Reset all data"""
    if not args.yes:
        print("Error: --yes flag is required to reset")
        print("This will delete all indexed data!")
        return

    engine = MemoryEngine()
    engine.clear_all()
    print("All data cleared")
    engine.close()


async def cmd_expand(args):
    """Expand a chunk (show full context)"""
    engine = MemoryEngine()

    # Get record by ID
    record = engine.db.get_by_id(args.chunk_id)

    if record:
        if args.json_output:
            import json

            print(json.dumps(record, indent=2))
        else:
            print(f"ID: {record['id']}")
            print(f"Source: {record['metadata'].get('source', 'Unknown')}")
            print(f"\n{record['text']}")
    else:
        print(f"Error: Chunk {args.chunk_id} not found")

    engine.close()


async def cmd_add(args):
    """Add a memory"""
    engine = MemoryEngine()

    metadata = {}
    if args.metadata:
        for pair in args.metadata:
            if "=" in pair:
                key, value = pair.split("=", 1)
                metadata[key] = value

    record_id = await engine.add_memory(args.text, metadata=metadata)
    print(f"Added memory: {record_id}")

    engine.close()


def main():
    parser = argparse.ArgumentParser(
        description="oG-Memory - openGauss-based semantic memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument(
        "--json-output", action="store_true", help="Output as JSON"
    )
    search_parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid (vector + BM25) search"
    )

    # Index command
    index_parser = subparsers.add_parser("index", help="Index markdown files")
    index_parser.add_argument("path", help="File or directory to index")
    index_parser.add_argument("--force", action="store_true", help="Force re-index")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset all data")
    reset_parser.add_argument("--yes", action="store_true", help="Confirm reset")

    # Expand command
    expand_parser = subparsers.add_parser("expand", help="Expand a chunk")
    expand_parser.add_argument("chunk_id", help="Chunk ID to expand")
    expand_parser.add_argument(
        "--json-output", action="store_true", help="Output as JSON"
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("text", help="Memory text")
    add_parser.add_argument("--metadata", nargs="*", help="Metadata as key=value pairs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run command
    if args.command == "search":
        asyncio.run(cmd_search(args))
    elif args.command == "index":
        asyncio.run(cmd_index(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))
    elif args.command == "reset":
        asyncio.run(cmd_reset(args))
    elif args.command == "expand":
        asyncio.run(cmd_expand(args))
    elif args.command == "add":
        asyncio.run(cmd_add(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
