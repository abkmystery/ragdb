from __future__ import annotations
import argparse
from .core import RAGdb


def main():
    parser = argparse.ArgumentParser(description="RAGdb CLI")
    parser.add_argument("--db", default="RAGdb.ragdb")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest").add_argument("path")
    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=5)
    sub.add_parser("delete").add_argument("path")

    args = parser.parse_args()
    db = RAGdb(args.db)

    if args.cmd == "ingest":
        # Simple heuristic to check if it's a file or folder
        if "." in args.path.split("/")[-1] and not args.path.endswith("/"):
            db.ingest_file(args.path)
        else:
            db.ingest_folder(args.path)

    elif args.cmd == "search":
        print(f"Searching for: '{args.query}'...")
        results = db.search(args.query, top_k=args.k)

        # FIX: Uses dot notation (.score, .path) for the SearchResult object
        for i, res in enumerate(results):
            print(f"[{i + 1}] Score: {res.score:.4f} | {res.media_type}")
            print(f"File: {res.path}")
            print(f"Snippet: {res.preview[:200]}...\n")

    elif args.cmd == "delete":
        db.delete_file(args.path)


if __name__ == "__main__":
    main()