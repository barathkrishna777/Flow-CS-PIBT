"""Compare evaluation results across all wave checkpoints.

Usage:
    python compare_results.py                       # auto-discover all CSVs in logs/
    python compare_results.py --csv logs/a.csv logs/b.csv   # specific files
"""
import csv
import argparse
import glob
import os
from collections import defaultdict


def load_results(csv_path):
    """Load CSV and return dict of (map, agents) -> list of at-goal percentages."""
    results = defaultdict(list)
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_name = row['mapName']
            agent_num = int(row['agentNum'])
            at_goal = int(row['num_agents_at_goal'])
            pct = at_goal / agent_num * 100
            results[(map_name, agent_num)].append(pct)

    # Average across scenarios
    return {k: sum(v) / len(v) for k, v in results.items()}


def format_delta(new, old):
    """Format a delta value with color indicator."""
    if old is None or new is None:
        return "  ---  "
    delta = new - old
    if delta > 0.5:
        return f"+{delta:5.1f}%"
    elif delta < -0.5:
        return f"{delta:5.1f}%"
    else:
        return f"  ={delta:+4.1f}%"


def main():
    parser = argparse.ArgumentParser()
    # Auto-discover all batch_results CSVs, sorted for consistent ordering
    default_csvs = sorted(glob.glob("logs/batch_results_*.csv"))
    if not default_csvs:
        default_csvs = ["logs/batch_results_wave2.csv"]  # fallback
    parser.add_argument("--csv", nargs="+", default=default_csvs)
    args = parser.parse_args()

    # Load all CSVs
    labels = []
    datasets = []
    for path in args.csv:
        name = os.path.basename(path).replace("batch_results_", "").replace(".csv", "")
        data = load_results(path)
        if data is not None:
            labels.append(name)
            datasets.append(data)
            print(f"Loaded: {path} ({len(data)} entries) -> '{name}'")
        else:
            print(f"Not found: {path} (skipping)")

    if not datasets:
        print("No data to compare!")
        return

    # Gather all (map, agents) keys
    all_keys = set()
    for d in datasets:
        all_keys.update(d.keys())

    maps = sorted(set(k[0] for k in all_keys))
    agents = sorted(set(k[1] for k in all_keys))

    # Print comparison table
    col_width = 12
    header_cols = [f"{l:>{col_width}}" for l in labels]
    delta_cols = [f"{'delta':>{col_width-2}}" for _ in labels[1:]] if len(labels) > 1 else []

    print(f"\n{'='*80}")
    print(f"  COMPARISON: At-Goal Rate (%) averaged across scenarios")
    print(f"{'='*80}")

    # Build header
    hdr = f"{'Map':>20s} {'Agents':>6s}"
    for l in labels:
        hdr += f" {l:>14s}"
    if len(labels) > 1:
        for l in labels[1:]:
            hdr += f" {'d('+l+')':>14s}"
    print(hdr)
    print("-" * len(hdr))

    baseline = datasets[0] if datasets else {}

    for map_name in maps:
        for agent_num in agents:
            key = (map_name, agent_num)
            row = f"{map_name:>20s} {agent_num:>6d}"

            vals = []
            for d in datasets:
                v = d.get(key)
                vals.append(v)
                if v is not None:
                    row += f" {v:>13.1f}%"
                else:
                    row += f" {'---':>14s}"

            # Deltas vs baseline
            if len(vals) > 1:
                for v in vals[1:]:
                    row += f" {format_delta(v, vals[0]):>14s}"

            print(row)
        print()

    # Summary
    print(f"{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    for i, (label, data) in enumerate(zip(labels, datasets)):
        all_vals = list(data.values())
        avg = sum(all_vals) / len(all_vals) if all_vals else 0
        full_success = sum(1 for v in all_vals if v == 100)
        print(f"  {label}: avg={avg:.1f}% | 100% scenarios: {full_success}/{len(all_vals)}")

        if i > 0 and baseline:
            # Compute average delta
            deltas = []
            for k, v in data.items():
                if k in baseline:
                    deltas.append(v - baseline[k])
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                print(f"    vs {labels[0]}: avg delta = {avg_delta:+.1f}%")


if __name__ == "__main__":
    main()
