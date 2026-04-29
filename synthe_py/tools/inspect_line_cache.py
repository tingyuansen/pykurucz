#!/usr/bin/env python3
"""Inspect parsed/compiled line cache artefacts quickly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthe_py.io.lines import compiler, parsed_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect line cache files")
    parser.add_argument("--catalog", type=Path, required=True, help="Path to gfall catalog")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory (defaults to <catalog_dir>/.py_line_cache)",
    )
    parser.add_argument("--wlbeg", type=float, default=368.0)
    parser.add_argument("--wlend", type=float, default=372.0)
    parser.add_argument("--resolution", type=float, default=300000.0)
    parser.add_argument("--line-filter", action="store_true", default=False)
    parser.add_argument(
        "--show-key-components",
        action="store_true",
        default=False,
        help="Include cache key payload components (logic versions, source fingerprint, params).",
    )
    args = parser.parse_args()

    parsed_paths = parsed_cache.cache_paths(args.catalog, cache_directory=args.cache_dir)
    parsed_manifest = parsed_cache.load_manifest(args.catalog, cache_directory=args.cache_dir)

    compiled_path = compiler._cache_file_path(  # type: ignore[attr-defined]
        catalog_path=args.catalog,
        wlbeg=args.wlbeg,
        wlend=args.wlend,
        resolution=args.resolution,
        line_filter=args.line_filter,
        cache_directory=args.cache_dir,
    )
    compiled_summary = compiler.compiled_cache_summary(compiled_path)
    parsed_payload = (
        parsed_cache.parsed_cache_key_payload(args.catalog)
        if args.show_key_components
        else None
    )
    compiled_payload = (
        compiler.compiled_cache_key_payload(
            catalog_path=args.catalog,
            wlbeg=args.wlbeg,
            wlend=args.wlend,
            resolution=args.resolution,
            line_filter=args.line_filter,
        )
        if args.show_key_components
        else None
    )

    output = {
        "parsed_cache": {
            "npz_path": str(parsed_paths.npz),
            "manifest_path": str(parsed_paths.manifest),
            "exists": parsed_paths.npz.exists(),
            "manifest": parsed_manifest,
            "key_payload": parsed_payload,
        },
        "compiled_cache": {
            "path": str(compiled_path),
            "exists": compiled_path.exists(),
            "summary": compiled_summary,
            "params": {
                "wlbeg": args.wlbeg,
                "wlend": args.wlend,
                "resolution": args.resolution,
                "line_filter": args.line_filter,
            },
            "key_payload": compiled_payload,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


