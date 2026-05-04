#!/usr/bin/env python3
"""Compute paired McNemar tests for matched FLOMAP-vs-SSIL evaluations."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from report_data import CSV_PATHS, TABLE_DIR, ensure_dirs, write_csv


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_success(path: Path) -> dict[tuple[str, str, int, int], bool]:
    rows: dict[tuple[str, str, int, int], bool] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["mapName"],
                row["scenFile"],
                int(row["agentNum"]),
                int(row.get("seed", 0)),
            )
            rows[key] = truthy(row["success"])
    return rows


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def binom_sf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def exact_mcnemar(flow: dict, ssil: dict, keys: list[tuple[str, str, int, int]]) -> dict[str, object]:
    flo_only = ssil_only = both = neither = 0
    for key in keys:
        flo = flow[key]
        sil = ssil[key]
        if flo and sil:
            both += 1
        elif flo and not sil:
            flo_only += 1
        elif (not flo) and sil:
            ssil_only += 1
        else:
            neither += 1
    discordant = flo_only + ssil_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2.0
            * min(
                binom_cdf(min(flo_only, ssil_only), discordant),
                binom_sf(max(flo_only, ssil_only), discordant),
            ),
        )
    return {
        "n": len(keys),
        "flomap_only": flo_only,
        "ssil_only": ssil_only,
        "both_success": both,
        "both_failure": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def main() -> int:
    ensure_dirs()
    flow = load_success(CSV_PATHS["rishi8_flow"])
    ssil = load_success(CSV_PATHS["rishi8_ssil"])
    keys = sorted(set(flow) & set(ssil))
    stress = {"random-32-32-10", "maze-128-128-2"}
    floor = {"warehouse-10-20-10-2-1"}
    slices = {
        "Full primary held-out evaluation": keys,
        "Coordination stress maps": [key for key in keys if key[0] in stress],
        "Dense random map": [key for key in keys if key[0] == "random-32-32-10"],
        "Maze bottleneck map": [key for key in keys if key[0] == "maze-128-128-2"],
        "Excluding coordination stress maps": [key for key in keys if key[0] not in stress],
        "Excluding stress and floor-case maps": [
            key for key in keys if key[0] not in stress | floor
        ],
    }
    rows = []
    for name, slice_keys in slices.items():
        result = exact_mcnemar(flow, ssil, slice_keys)
        result["slice"] = name
        result["p_display"] = "<0.001" if result["p_value"] < 0.001 else f"{result['p_value']:.4f}"
        rows.append(result)

    write_csv(
        TABLE_DIR / "paired_mcnemar_tests.csv",
        rows,
        [
            "slice",
            "n",
            "flomap_only",
            "ssil_only",
            "both_success",
            "both_failure",
            "discordant",
            "p_value",
            "p_display",
        ],
    )
    print(f"Wrote paired McNemar tests to {TABLE_DIR / 'paired_mcnemar_tests.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
