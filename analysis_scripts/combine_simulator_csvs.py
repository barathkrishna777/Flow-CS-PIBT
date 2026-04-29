"""Combine simulator CSV shards into one simulator-format CSV.

The grid summarizer treats every input CSV as a separate logical run. Use this
helper after sharded evals to concatenate shard CSVs with a single header, then
pass the combined CSV to ``analysis_scripts.summarize_grid_eval``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def combine_csvs(inputs: list[str], output: str, expect_rows: int | None = None) -> int:
    if not inputs:
        raise ValueError("at least one input CSV is required")

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    header: list[str] | None = None
    total_rows = 0

    with open(output, "w", newline="") as out_f:
        writer = csv.writer(out_f)

        for path in inputs:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

            with open(path, newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    current_header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"empty CSV: {path}") from exc

                if header is None:
                    header = current_header
                    writer.writerow(header)
                elif current_header != header:
                    raise ValueError(
                        f"header mismatch in {path}\n"
                        f"expected: {header}\n"
                        f"got:      {current_header}"
                    )

                shard_rows = 0
                for row in reader:
                    if not row:
                        continue
                    writer.writerow(row)
                    shard_rows += 1

            total_rows += shard_rows
            print(f"{path}: {shard_rows} rows")

    print(f"Wrote {total_rows} rows to {output}")

    if expect_rows is not None and total_rows != expect_rows:
        print(
            f"ERROR: expected {expect_rows} rows, got {total_rows}",
            file=sys.stderr,
        )
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine simulator CSV shards")
    parser.add_argument("inputs", nargs="+", help="Shard CSV files")
    parser.add_argument("-o", "--output", required=True, help="Combined CSV path")
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=None,
        help="Fail if the combined data row count differs from this value",
    )
    args = parser.parse_args()

    try:
        return combine_csvs(args.inputs, args.output, args.expect_rows)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
