#!/usr/bin/env python3
"""Extract numerical tables embedded in `atlas7v.for`.

Kurucz' atlas7v source keeps several large DATA blocks (partition-function
coefficients, energy level lists, scaling helpers, …) encoded in base-37
integers.  This tool decodes those blocks once and writes them into a handy
NumPy archive so the Python port can consume the exact same constants without
re-implementing the legacy reader.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np


ALPHA_RE = re.compile(r"[A-Za-z]")
LABEL_TOKENS = {"D+F", "AEL", "FAK", "G"}


def _split_tokens(raw_block: str) -> list[str]:
    """Split a DATA block into individual tokens, preserving order."""
    cleaned = raw_block.replace("\n", " ")
    # Fortran allows trailing comments via '!'; strip anything after it.
    if "!" in cleaned:
        cleaned = " ".join(part.split("!")[0] for part in cleaned.split("!"))
    tokens = re.split(r"[,\s]+", cleaned.strip())
    return [token for token in tokens if token]


def _parse_numeric_tokens(tokens: Iterable[str]) -> list[float]:
    """Parse tokens, skipping mnemonic annotations (D+F, AEL, G, …)."""
    values: list[float] = []
    skip_numeric = False
    for token in tokens:
        if skip_numeric:
            # Skip the numeric part that accompanies the mnemonic label we just saw.
            skip_numeric = False
            continue

        # Fortran fixed-form continuation puts a digit in column 6; when we
        # strip whitespace, that digit can cling to the first value (e.g.
        # "1-4254").  Peel off the leading digits if they precede a signed literal.
        cont_match = re.match(r"^(\d+)([-+].*)$", token)
        if cont_match:
            token = cont_match.group(2)

        # Drop bare line continuation digits ("1", "2", …) that remain after stripping.
        if token.isdigit() and len(token) <= 2:
            continue

        repl = token.replace("D", "E")
        if "*" in repl:
            count_str, value_str = token.split("*", 1)
            try:
                count = int(count_str)
            except ValueError:
                # Treat malformed repetition (e.g. kw*1.) as a label.
                skip_numeric = True
                continue
            value_str = value_str.strip()
            if not value_str:
                raise ValueError(f"Missing value in repetition token '{token}'")
            try:
                value = float(value_str.replace("D", "E"))
            except ValueError:
                skip_numeric = True
                continue
            values.extend([value] * count)
            continue
        if not repl:
            continue
        try:
            values.append(float(repl))
        except ValueError as exc:
            upper = token.upper()
            if upper in LABEL_TOKENS or ALPHA_RE.fullmatch(upper):
                skip_numeric = True
                continue
            # Ignore stray label tokens.
            continue
    return values


def parse_data_blocks(source: Path) -> tuple[dict[str, list[float]], str]:
    """Return a mapping of DATA block name -> list of numeric values and raw text."""
    text = source.read_text()
    lines = text.splitlines()
    blocks: dict[str, list[float]] = {}

    i = 0
    total_lines = len(lines)
    data_re = re.compile(r"^\s*DATA\s+([A-Za-z0-9]+)\s*/?(.*)$", re.IGNORECASE)

    while i < total_lines:
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("C", "c")):
            i += 1
            continue
        match = data_re.match(line)
        if not match:
            i += 1
            continue

        name = match.group(1).upper()
        line_content = line
        if "!" in line_content:
            line_content = line_content.split("!")[0]

        first_slash = line_content.find("/")
        if first_slash == -1:
            i += 1
            continue

        pending = line_content[first_slash + 1 :]
        segments: list[str] = []
        j = i

        while True:
            current = pending.strip()
            if current:
                if "/" in current:
                    before, _, tail = current.partition("/")
                    segments.append(before)
                    pending = tail
                    break
                segments.append(current)
            j += 1
            if j >= total_lines:
                pending = ""
                break
            next_line = lines[j]
            if not next_line.strip() or next_line.lstrip().startswith(("C", "c")):
                continue
            pending = next_line
            if "!" in pending:
                pending = pending.split("!")[0]

        tokens = _split_tokens(" ".join(segments))
        if tokens:
            blocks[name] = _parse_numeric_tokens(tokens)

        i = max(j, i) + 1

    return blocks, text


CHUNK_SEQUENCE = [
    "NNN01",
    "NNN02",
    "NNN03",
    "NNN04",
    "NNN05",
    "NNN06",
    "NNN07",
    "NNN08",
    "NNN09",
    "NNN10",
    "NNN11",
    "NNN12",
    "NNN13",
    "NNN14",
    "NNN15",
    "NNN16",
    "NNN17",
    "NNN18",
    "NNN19",
    "NNN20",
    "NNN21",
    "NNN22",
    "NNN23",
    "NNN24",
    "NNN25",
    "NNN26",
    "NNN27",
    "NNN28",
    "NNN29",
    "NNN30",
    "NNN31",
    "NNN32",
    "NNN33",
    "NNN34",
    "NNN35",
    "NNN36",
    "NNN37",
    "NNN38",
    "NNN39",
    "NNN40",
    "NNN67",
    "NNN88",
]


def assemble_nnn(blocks: dict[str, list[float]]) -> np.ndarray:
    """Combine the segmented NNN## arrays into the full (6, 374) integer table."""
    nnn_values: list[int] = []
    for name in CHUNK_SEQUENCE:
        if name not in blocks:
            raise KeyError(f"Missing {name} in atlas7v DATA blocks")
        nnn_values.extend(int(round(val)) for val in blocks[name])
    if len(nnn_values) % 6 != 0:
        raise ValueError("Combined NNN data length is not divisible by 6")
    # Fortran DATA statements populate column-major order.
    nnn_array = np.array(nnn_values, dtype=np.int64).reshape((6, -1), order="F")
    if nnn_array.shape[1] != 374:
        raise ValueError(f"Unexpected NNN column count: {nnn_array.shape[1]}")
    return nnn_array


FLOAT_TABLES = [
    "EHYD",
    "GHYD",
    "EHE1",
    "GHE1",
    "EHE2",
    "GHE2",
    "EC1",
    "GC1",
    "EMG1",
    "GMG1",
    "EMG2",
    "GMG2",
    "EAL1",
    "GAL1",
    "ESI1",
    "GSI1",
    "ESI2",
    "GSI2",
    "EK1",
    "GK1",
    "ECA1",
    "GCA1",
    "ECA2",
    "GCA2",
    "LOCZ",
    "SCALE",
]


def build_output_tables(blocks: dict[str, list[float]], text: str) -> dict[str, np.ndarray]:
    """Translate the parsed lists into structured NumPy arrays."""
    tables: dict[str, np.ndarray] = {}
    try:
        tables["NNN"] = assemble_nnn(blocks)
    except (KeyError, ValueError) as exc:
        print(f"Warning: skipping NNN table ({exc})")

    for name in FLOAT_TABLES:
        data = blocks.get(name)
        if data is None:
            match = re.search(rf"DATA\s+{name}\s*/(.*?)/", text, re.IGNORECASE | re.DOTALL)
            if not match:
                raise KeyError(f"Missing DATA statement for {name}")
            tokens = _split_tokens(match.group(1))
            data = _parse_numeric_tokens(tokens)
        array = np.array(data, dtype=np.float64)
        if name == "LOCZ":
            array = array.astype(np.int64)
        tables[name] = array
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode atlas7v DATA tables.")
    parser.add_argument(
        "source",
        type=Path,
        help="Path to atlas7v.for (default: kurucz/src/atlas7v.for)",
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "src" / "atlas7v.for",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination .npz file (default: synthe_py/data/atlas_tables.npz)",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "data" / "atlas_tables.npz",
    )
    args = parser.parse_args()

    blocks, text = parse_data_blocks(args.source)
    tables = build_output_tables(blocks, text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **tables)

    print(f"Wrote {len(tables)} tables to {args.output}")


if __name__ == "__main__":
    main()

