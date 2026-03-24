"""Evaluate on the full 27-map MAPF benchmark (matching Rishi's paper).

Usage:
    python eval_full.py <model_path>                   # all 27 maps
    python eval_full.py <model_path> --test-only        # 8 held-out test maps only
    python eval_full.py <model_path> --maps den312d empty-48-48  # specific maps
"""
import os, sys, subprocess, glob, argparse

# ── The 8 held-out test maps (never seen during training) ──
HELD_OUT_TEST = [
    "Paris_1_256", "empty-48-48", "maze-128-128-2", "random-64-64-10",
    "random-32-32-10", "warehouse-10-20-10-2-1", "den312d", "den520d",
]

# ── 4 maps omitted from the paper (pipeline issues) ──
OMITTED = {"brc202d", "orz900d", "maze-128-128-1", "maze-128-128-10"}

# ── All 33 MovingAI MAPF benchmark maps ──
ALL_MAPS = [
    "Berlin_1_256", "Boston_0_256", "Paris_1_256",
    "brc202d", "den312d", "den520d",
    "empty-8-8", "empty-16-16", "empty-32-32", "empty-48-48",
    "ht_chantry", "ht_mansion_n", "lak303d", "lt_gallowstemplar_n",
    "maze-128-128-1", "maze-128-128-2", "maze-128-128-10",
    "maze-32-32-2", "maze-32-32-4",
    "orz900d", "ost003d",
    "random-32-32-10", "random-32-32-20", "random-64-64-10", "random-64-64-20",
    "room-32-32-4", "room-64-64-16", "room-64-64-8",
    "w_woundedcoast",
    "warehouse-10-20-10-2-1", "warehouse-10-20-10-2-2",
    "warehouse-20-40-10-2-1", "warehouse-20-40-10-2-2",
]

# Paper uses agent increments of 100
AGENT_INCREMENT = 100
SCENARIOS_PER_MAP = 25  # all 25 random scenarios

MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs/large_scale"
SCEN_DIR = "data/scen-random"


def get_max_agents(map_name):
    """Return max agents available in scenario files for this map."""
    scens = glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-1.scen"))
    if not scens:
        return 0
    with open(scens[0]) as f:
        lines = f.readlines()
    return len(lines) - 1  # subtract header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model checkpoint")
    parser.add_argument("--output", default=None, help="Output CSV path (default: logs/eval_full_<model>.csv)")
    parser.add_argument("--test-only", action="store_true", help="Only evaluate on 8 held-out test maps")
    parser.add_argument("--maps", nargs="*", default=None, help="Specific maps to evaluate")
    parser.add_argument("--max-agents", type=int, default=1000, help="Max agents to test (default: 1000)")
    parser.add_argument("--scenarios", type=int, default=SCENARIOS_PER_MAP, help="Scenarios per map (default: 25)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: {args.model} not found")
        sys.exit(1)

    # Select maps
    if args.maps:
        test_maps = args.maps
    elif args.test_only:
        test_maps = HELD_OUT_TEST
    else:
        test_maps = [m for m in ALL_MAPS if m not in OMITTED]

    # Output path
    if args.output:
        csv_path = args.output
    else:
        model_tag = os.path.basename(args.model).replace(".pt", "")
        csv_path = f"logs/eval_full_{model_tag}.csv"

    os.makedirs("logs", exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    import torch
    use_gpu = torch.cuda.is_available()

    # Build run list
    runs = []
    for map_name in test_maps:
        scens = sorted(glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")))[:args.scenarios]
        if not scens:
            print(f"Skipping {map_name} - no scenario files")
            continue

        max_avail = get_max_agents(map_name)
        agent_counts = list(range(AGENT_INCREMENT, min(max_avail, args.max_agents) + 1, AGENT_INCREMENT))
        if not agent_counts:
            agent_counts = [max_avail]  # map has fewer than 100 agents

        for scen in scens:
            bn = os.path.basename(scen).replace(".scen", "")
            bd = os.path.join(BD_DIR, f"{bn}_bds.npz")
            if not os.path.exists(bd):
                continue
            for n in agent_counts:
                runs.append((map_name, scen, bd, n))

    total = len(runs)
    is_test = {m for m in HELD_OUT_TEST}
    print(f"{'='*60}")
    print(f"Full Benchmark Evaluation: {args.model}")
    print(f"Maps: {len(test_maps)} | Scenarios/map: {args.scenarios} | GPU: {use_gpu}")
    print(f"Total runs: {total}")
    print(f"Output: {csv_path}")
    print(f"{'='*60}\n")

    for i, (map_name, scen, bd, n) in enumerate(runs, 1):
        tag = " [TEST]" if map_name in is_test else ""
        print(f"[{i}/{total}] {map_name}{tag} | {n} agents")
        subprocess.run([
            "python", "-m", "main_pys.simulator",
            f"--mapNpzFile={MAP_NPZ}", f"--mapName={map_name}",
            f"--scenFile={scen}", f"--bdNpzFile={bd}",
            f"--modelPath={args.model}", f"--outputCSVFile={csv_path}",
            "--maxSteps=5x", f"--seed={args.seed}",
            f"--useGPU={'True' if use_gpu else 'False'}",
            f"--agentNum={n}", "--shieldType=CS-PIBT"
        ])

    print(f"\nDone! Results: {csv_path}")


if __name__ == "__main__":
    main()
