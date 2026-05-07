"""Compare CS-PIBT, Lattice-PIBT, and Lattice-Cardinal CS-PIBT.

Three conditions evaluated on the same model, maps, and scenarios:
  1. CS-PIBT           — standard flow → 5-cardinal preferences → 1-step PIBT
  2. Lattice-PIBT      — flow → 17-primitive lattice PIBT (multi-step)
  3. Lattice-Cardinal  — flow → 17-prim scores max-pooled to 5 cardinals → CS-PIBT

Usage:
    python eval_lattice.py data/model/large_scale_flow_best.pt
    python eval_lattice.py data/model/large_scale_flow_best.pt --extended
    python eval_lattice.py --model data/model/YOUR_CHECKPOINT.pt
"""
import os, sys, subprocess, argparse, csv

MAP_NPZ   = "data/all_maps.npz"
BD_DIR    = "data/bd_npzs/large_scale"
SCEN_DIR  = "data/mapf-scen-random"

DEFAULT_MAPS = [
    ("empty-48-48",            "empty-48-48-random-1.scen"),
    ("random-32-32-10",        "random-32-32-10-random-1.scen"),
    ("den312d",                "den312d-random-1.scen"),
    ("Paris_1_256",            "Paris_1_256-random-1.scen"),
    ("warehouse-10-20-10-2-1", "warehouse-10-20-10-2-1-random-1.scen"),
]
DEFAULT_AGENTS = [100, 200, 400]
NUM_STEPS = 5
CONSENSUS = 3
TAU       = 0.3
WAIT_THRESH = 0.25
MAX_STEPS = "3x"
TIME_LIMIT = 120


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--model", "-m", dest="model_opt", default=None)
    parser.add_argument("--agents", nargs="*", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--extended", action="store_true",
                        help="Also test 800 agents")
    parser.add_argument("--speed-bonus", type=float, default=0.3,
                        help="Additive speed bonus for double-step lattice primitives (default 0.3)")
    parser.add_argument("--output-dir", default="evals/lattice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--skip-lattice-pibt", action="store_true",
                        help="Skip the multi-step Lattice-PIBT condition (useful when it times out)")
    args = parser.parse_args()

    args.model = args.model_opt or args.model
    if not args.model:
        parser.error("Provide model path as positional arg or --model/-m")
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    agent_counts = list(args.agents)
    if args.extended:
        agent_counts = sorted(set(agent_counts) | {800})

    os.makedirs(args.output_dir, exist_ok=True)
    csv_cs    = os.path.join(args.output_dir, "cs_pibt.csv")
    csv_lat   = os.path.join(args.output_dir, "lattice_pibt.csv")
    csv_lc    = os.path.join(args.output_dir, "lattice_cardinal_pibt.csv")

    import torch
    use_gpu = torch.cuda.is_available()

    # ── Build run list ──────────────────────────────────────────────────────
    runs = []
    for map_name, scen_file in DEFAULT_MAPS:
        scen_path = os.path.join(SCEN_DIR, scen_file)
        bn        = scen_file.replace(".scen", "")
        bd_path   = os.path.join(BD_DIR, f"{bn}_bds.npz")
        if not os.path.exists(scen_path):
            print(f"WARNING: missing scen {scen_path}, skipping {map_name}")
            continue
        if not os.path.exists(bd_path):
            print(f"WARNING: missing BD {bd_path}, skipping {map_name}")
            continue
        with open(scen_path) as f:
            max_avail = len(f.readlines()) - 1
        for n in agent_counts:
            if n > max_avail:
                print(f"WARNING: {map_name} has {max_avail} agents, skipping {n}")
                continue
            runs.append((map_name, scen_path, bd_path, n))

    common_flags = [
        f"--mapNpzFile={MAP_NPZ}",
        f"--modelPath={args.model}",
        f"--maxSteps={MAX_STEPS}",
        f"--seed={args.seed}",
        f"--useGPU={'True' if use_gpu else 'False'}",
        f"--numIntegrationSteps={NUM_STEPS}",
        f"--tau={TAU}",
        f"--waitThreshold={WAIT_THRESH}",
        f"--numConsensusSamples={CONSENSUS}",
        f"--timeLimit={TIME_LIMIT}",
        f"--hiddenDim={args.hidden_dim}",
        f"--numLayers={args.num_layers}",
        "--policyType=flow",
    ]

    num_modes = 3 if not args.skip_lattice_pibt else 2
    total = len(runs)
    print(f"{'='*65}")
    print(f"CS-PIBT vs Lattice-Cardinal vs Lattice-PIBT comparison")
    print(f"Model  : {args.model}")
    print(f"Maps   : {len(DEFAULT_MAPS)} | Agents: {agent_counts} | Steps: {NUM_STEPS}")
    print(f"GPU    : {use_gpu} | Total runs: {total * num_modes} ({total} per condition)")
    print(f"Output : {args.output_dir}/")
    print(f"{'='*65}\n")

    def run_one(map_name, scen_path, bd_path, n, mode, out_csv):
        """Run a single simulator call.

        mode is one of: "CS-PIBT", "Lattice-PIBT", "Lattice-Cardinal"
        """
        tmp = f"logs/_eval_lattice_tmp_{mode.replace(' ', '_').replace('-', '_')}.csv"
        os.makedirs("logs", exist_ok=True)
        if os.path.exists(tmp):
            os.remove(tmp)

        if mode == "CS-PIBT":
            shield_flag = "--shieldType=CS-PIBT"
            extra = []
        elif mode == "Lattice-PIBT":
            shield_flag = "--shieldType=Lattice-PIBT"
            extra = [
                "--latticeScoreMode=velocity",
                f"--latticeSpeedBonus={args.speed_bonus}",
            ]
        else:  # Lattice-Cardinal
            shield_flag = "--shieldType=CS-PIBT"
            extra = [
                "--latticeCardinalMode=True",
                f"--latticeSpeedBonus={args.speed_bonus}",
            ]

        cmd = (
            ["python", "-m", "main_pys.simulator"]
            + common_flags
            + [
                f"--mapName={map_name}",
                f"--scenFile={scen_path}",
                f"--bdNpzFile={bd_path}",
                f"--agentNum={n}",
                shield_flag,
                f"--outputCSVFile={tmp}",
            ]
            + extra
        )

        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIME_LIMIT + 60)
        except subprocess.TimeoutExpired:
            print(f"  {mode} TIMEOUT")
            return None

        if not os.path.exists(tmp):
            print(f"  {mode} NO OUTPUT")
            return None

        with open(tmp) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        row = rows[0]
        at_goal  = int(row.get("num_agents_at_goal", 0))
        success  = row.get("success", "False") == "True"
        runtime  = float(row.get("runtime", 0))
        cost     = row.get("total_cost_true", 0)
        cost_nr  = row.get("total_cost_not_resting_at_goal", 0)
        pct      = 100.0 * at_goal / n

        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["shield","map","agents","at_goal_pct","success",
                                  "runtime","total_cost","cost_not_resting"])
            writer.writerow([mode, map_name, n, f"{pct:.1f}", success,
                              f"{runtime:.2f}", cost, cost_nr])

        return at_goal, success, runtime, pct

    # ── Run all conditions ───────────────────────────────────────────────────
    for i, (map_name, scen_path, bd_path, n) in enumerate(runs, 1):
        print(f"[{i}/{total}] {map_name} | {n} agents")

        r_cs = run_one(map_name, scen_path, bd_path, n, "CS-PIBT", csv_cs)
        r_lc = run_one(map_name, scen_path, bd_path, n, "Lattice-Cardinal", csv_lc)

        if not args.skip_lattice_pibt:
            r_lat = run_one(map_name, scen_path, bd_path, n, "Lattice-PIBT", csv_lat)
        else:
            r_lat = None

        if r_cs:
            print(f"  CS-PIBT          : {r_cs[0]}/{n} ({r_cs[3]:.1f}%)  {r_cs[2]:.1f}s")
        if r_lc:
            gain = r_lc[3] - (r_cs[3] if r_cs else 0)
            sign = "+" if gain >= 0 else ""
            print(f"  Lattice-Cardinal : {r_lc[0]}/{n} ({r_lc[3]:.1f}%)  {r_lc[2]:.1f}s  [{sign}{gain:.1f}%]")
        if r_lat:
            gain = r_lat[3] - (r_cs[3] if r_cs else 0)
            sign = "+" if gain >= 0 else ""
            print(f"  Lattice-PIBT     : {r_lat[0]}/{n} ({r_lat[3]:.1f}%)  {r_lat[2]:.1f}s  [{sign}{gain:.1f}%]")
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print("SUMMARY — avg agents-at-goal % by agent count")
    print(f"{'='*65}")
    try:
        import pandas as pd
        entries = [("CS-PIBT", csv_cs), ("Lattice-Cardinal", csv_lc)]
        if not args.skip_lattice_pibt:
            entries.append(("Lattice-PIBT", csv_lat))
        for label, path in entries:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["at_goal_pct"] = pd.to_numeric(df["at_goal_pct"], errors="coerce")
                avg = df.groupby("agents")["at_goal_pct"].mean()
                print(f"\n{label}:")
                print(avg.to_string())
    except ImportError:
        print("  (install pandas for summary)")

    print(f"\nDone!")
    print(f"  CS-PIBT          : {csv_cs}")
    print(f"  Lattice-Cardinal : {csv_lc}")
    if not args.skip_lattice_pibt:
        print(f"  Lattice-PIBT     : {csv_lat}")


if __name__ == "__main__":
    main()
