import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_command(cmd: Sequence[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def read_csv_rows(csv_paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    return rows


def build_checkpoint_name(policy: str, run_name: str) -> str:
    return f"continuous_{policy}_{run_name}_best.pt"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_main(rows: List[Dict[str, str]], output_dir: Path) -> None:
    metrics = [
        "agent_fraction_at_goal",
        "success",
        "runtime",
        "path_length_ratio",
        "smoothness",
        "collisions",
        "near_collisions",
        "obstacle_hits",
    ]
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["policy"], int(row["agents"]))
        for metric in metrics:
            grouped[key][metric].append(float(row[metric]))

    summary_rows = []
    for (policy, agents), metric_values in sorted(grouped.items()):
        summary_row: Dict[str, object] = {"policy": policy, "agents": agents}
        for metric in metrics:
            avg, std = mean_std(metric_values.get(metric, []))
            summary_row[f"{metric}_mean"] = avg
            summary_row[f"{metric}_std"] = std
        summary_rows.append(summary_row)

    summary_fields = ["policy", "agents"] + [f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "std")]
    write_csv(output_dir / "main_summary.csv", summary_fields, summary_rows)

    policies = sorted({row["policy"] for row in summary_rows})
    agent_counts = sorted({int(row["agents"]) for row in summary_rows})
    main_table_rows = []
    for policy in policies:
        row: Dict[str, object] = {"policy": policy}
        for agents in agent_counts:
            match = next((entry for entry in summary_rows if entry["policy"] == policy and entry["agents"] == agents), None)
            if match is None:
                row[f"{agents}_mean"] = ""
                row[f"{agents}_std"] = ""
            else:
                row[f"{agents}_mean"] = match["agent_fraction_at_goal_mean"]
                row[f"{agents}_std"] = match["agent_fraction_at_goal_std"]
        main_table_rows.append(row)

    main_fields = ["policy"] + [field for agents in agent_counts for field in (f"{agents}_mean", f"{agents}_std")]
    write_csv(output_dir / "main_table_agent_fraction_at_goal.csv", main_fields, main_table_rows)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    for metric, ylabel, filename in [
        ("agent_fraction_at_goal", "Agent Fraction At Goal", "scaling_agent_fraction_at_goal.png"),
        ("runtime", "Runtime (s)", "runtime_scaling.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for policy in policies:
            policy_rows = sorted(
                [entry for entry in summary_rows if entry["policy"] == policy],
                key=lambda entry: int(entry["agents"]),
            )
            xs = [int(entry["agents"]) for entry in policy_rows]
            ys = [float(entry[f"{metric}_mean"]) for entry in policy_rows]
            yerr = [float(entry[f"{metric}_std"]) for entry in policy_rows]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=2, label=policy)
        plt.xlabel("Agents")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()


def summarize_consensus(rows: List[Dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return

    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    runtime_grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["num_consensus_samples"]), int(row["agents"]))
        grouped[key].append(float(row["agent_fraction_at_goal"]))
        runtime_grouped[key].append(float(row["runtime"]))

    summary_rows = []
    consensus_values = sorted({key[0] for key in grouped})
    agent_counts = sorted({key[1] for key in grouped})
    for consensus, agents in sorted(grouped):
        goal_mean, goal_std = mean_std(grouped[(consensus, agents)])
        runtime_mean, runtime_std = mean_std(runtime_grouped[(consensus, agents)])
        summary_rows.append(
            {
                "num_consensus_samples": consensus,
                "agents": agents,
                "agent_fraction_at_goal_mean": goal_mean,
                "agent_fraction_at_goal_std": goal_std,
                "runtime_mean": runtime_mean,
                "runtime_std": runtime_std,
            }
        )

    write_csv(
        output_dir / "consensus_sweep_summary.csv",
        [
            "num_consensus_samples",
            "agents",
            "agent_fraction_at_goal_mean",
            "agent_fraction_at_goal_std",
            "runtime_mean",
            "runtime_std",
        ],
        summary_rows,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    for metric, ylabel, filename in [
        ("agent_fraction_at_goal", "Agent Fraction At Goal", "consensus_sweep_agent_fraction_at_goal.png"),
        ("runtime", "Runtime (s)", "consensus_sweep_runtime.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for consensus in consensus_values:
            subset = sorted(
                [row for row in summary_rows if int(row["num_consensus_samples"]) == consensus],
                key=lambda row: int(row["agents"]),
            )
            xs = [int(row["agents"]) for row in subset]
            ys = [float(row[f"{metric}_mean"]) for row in subset]
            yerr = [float(row[f"{metric}_std"]) for row in subset]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=2, label=f"c={consensus}")
        plt.xlabel("Agents")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()


def phase_a_run_name(prefix: str, seed: int) -> str:
    return f"{prefix}_s{seed}"


def stage_generate(args: argparse.Namespace, python_bin: str) -> None:
    cmd = [
        python_bin,
        str(REPO_ROOT / "generate_continuous_data.py"),
        "--map-dir",
        args.map_dir,
        "--scen-dir",
        args.scen_dir,
        "--maps",
        *args.maps,
        "--agent-counts",
        *[str(agent_count) for agent_count in args.agent_counts],
        "--output-dir",
        args.data_dir,
        "--expert-source",
        args.expert_source,
        "--max-scenarios",
        str(args.max_scenarios),
        "--rollout-horizon",
        str(args.rollout_horizon),
        "--dt",
        str(args.dt),
        "--max-speed",
        str(args.max_speed),
        "--agent-radius",
        str(args.agent_radius),
        "--goal-tolerance",
        str(args.goal_tolerance),
        "--wait-threshold",
        str(args.wait_threshold),
        "--num-directions",
        str(args.num_directions),
        "--suboptimality",
        str(args.suboptimality),
        "--time-limit",
        str(args.time_limit),
    ]
    if args.eecbs_repo:
        cmd.extend(["--eecbs-repo", args.eecbs_repo])
    if args.eecbs_binary:
        cmd.extend(["--eecbs-binary", args.eecbs_binary])
    run_command(cmd, args.dry_run)


def stage_train(args: argparse.Namespace, python_bin: str) -> None:
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    for policy in args.train_policies:
        for seed in args.seeds:
            run_name = phase_a_run_name(args.run_prefix, seed)
            checkpoint_path = Path(args.checkpoint_dir) / build_checkpoint_name(policy, run_name)
            if args.skip_existing and checkpoint_path.exists():
                print(f"Skipping existing checkpoint: {checkpoint_path}")
                continue
            cmd = [
                python_bin,
                "-m",
                "main_pys.train_continuous",
                "--data-dir",
                args.data_dir,
                "--map-dir",
                args.map_dir,
                "--policy-type",
                policy,
                "--run-name",
                run_name,
                "--output-dir",
                args.checkpoint_dir,
                "--epochs",
                str(args.epochs),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--k",
                str(args.k),
                "--m",
                str(args.m),
                "--num-directions",
                str(args.num_directions),
                "--wait-threshold",
                str(args.wait_threshold),
                "--max-speed",
                str(args.max_speed),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--val-split",
                str(args.val_split),
                "--seed",
                str(seed),
            ]
            if args.batch_size > 0:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers > 0:
                cmd.extend(["--num-workers", str(args.num_workers)])
            if args.no_weighted_sampling:
                cmd.append("--no-weighted-sampling")
            if args.cpu:
                cmd.append("--cpu")
            run_command(cmd, args.dry_run)


def build_eval_command(
    args: argparse.Namespace,
    python_bin: str,
    output_csv: Path,
    policy: str,
    model_path: Optional[Path],
    run_name: str,
    train_seed: Optional[int],
    num_consensus_samples: int,
    viz_dir: Optional[Path] = None,
    max_scenarios: Optional[int] = None,
    agent_counts: Optional[Sequence[int]] = None,
) -> List[str]:
    cmd = [
        python_bin,
        str(REPO_ROOT / "eval_continuous.py"),
        "--map-dir",
        args.map_dir,
        "--scen-dir",
        args.scen_dir,
        "--maps",
        *args.maps,
        "--agent-counts",
        *[str(agent_count) for agent_count in (agent_counts or args.agent_counts)],
        "--max-scenarios",
        str(max_scenarios or args.max_scenarios),
        "--policy",
        policy,
        "--run-name",
        run_name,
        "--eval-seed",
        str(args.eval_seed),
        "--output-csv",
        str(output_csv),
        "--shield-type",
        args.shield_type,
        "--num-integration-steps",
        str(args.num_integration_steps),
        "--num-consensus-samples",
        str(num_consensus_samples),
        "--tau",
        str(args.tau),
        "--k",
        str(args.k),
        "--m",
        str(args.m),
        "--max-steps",
        str(args.max_eval_steps),
        "--dt",
        str(args.dt),
        "--max-speed",
        str(args.max_speed),
        "--agent-radius",
        str(args.agent_radius),
        "--goal-tolerance",
        str(args.goal_tolerance),
    ]
    if model_path is not None:
        cmd.extend(["--model-path", str(model_path)])
    if train_seed is not None:
        cmd.extend(["--train-seed", str(train_seed)])
    if viz_dir is not None:
        cmd.extend(["--viz-dir", str(viz_dir)])
    if args.cpu:
        cmd.append("--cpu")
    return cmd


def stage_eval(args: argparse.Namespace, python_bin: str) -> None:
    eval_main_dir = Path(args.benchmark_dir) / "evals" / "main"
    eval_sweep_dir = Path(args.benchmark_dir) / "evals" / "sweeps"
    viz_dir = Path(args.benchmark_dir) / "viz"
    eval_main_dir.mkdir(parents=True, exist_ok=True)
    if args.enable_consensus_sweep:
        eval_sweep_dir.mkdir(parents=True, exist_ok=True)

    orca_csv = eval_main_dir / "orca.csv"
    if not args.skip_existing or not orca_csv.exists():
        run_command(
            build_eval_command(
                args,
                python_bin,
                output_csv=orca_csv,
                policy="orca",
                model_path=None,
                run_name=f"{args.run_prefix}_orca",
                train_seed=None,
                num_consensus_samples=1,
            ),
            args.dry_run,
        )

    for policy in args.train_policies:
        for seed in args.seeds:
            run_name = phase_a_run_name(args.run_prefix, seed)
            checkpoint_path = Path(args.checkpoint_dir) / build_checkpoint_name(policy, run_name)
            if not checkpoint_path.exists() and not args.dry_run:
                raise FileNotFoundError(f"Missing checkpoint for evaluation: {checkpoint_path}")

            output_csv = eval_main_dir / f"{policy}_seed{seed}.csv"
            if not args.skip_existing or not output_csv.exists():
                run_command(
                    build_eval_command(
                        args,
                        python_bin,
                        output_csv=output_csv,
                        policy=policy,
                        model_path=checkpoint_path,
                        run_name=f"{args.run_prefix}_{policy}_main",
                        train_seed=seed,
                        num_consensus_samples=args.default_consensus_samples if policy == "flow" else 1,
                    ),
                    args.dry_run,
                )

            if policy == "flow" and args.enable_consensus_sweep:
                for consensus in args.consensus_sweep:
                    if consensus == args.default_consensus_samples:
                        continue
                    sweep_csv = eval_sweep_dir / f"flow_c{consensus}_seed{seed}.csv"
                    if args.skip_existing and sweep_csv.exists():
                        print(f"Skipping existing sweep eval: {sweep_csv}")
                        continue
                    run_command(
                        build_eval_command(
                            args,
                            python_bin,
                            output_csv=sweep_csv,
                            policy="flow",
                            model_path=checkpoint_path,
                            run_name=f"{args.run_prefix}_flow_consensus_sweep",
                            train_seed=seed,
                            num_consensus_samples=consensus,
                        ),
                        args.dry_run,
                    )

    if args.make_viz:
        viz_main_dir = viz_dir / "main"
        viz_main_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            build_eval_command(
                args,
                python_bin,
                output_csv=viz_main_dir / "orca_viz.csv",
                policy="orca",
                model_path=None,
                run_name=f"{args.run_prefix}_orca_viz",
                train_seed=None,
                num_consensus_samples=1,
                viz_dir=viz_main_dir / "orca",
                max_scenarios=args.viz_max_scenarios,
                agent_counts=args.viz_agent_counts,
            ),
            args.dry_run,
        )
        if "flow" in args.train_policies:
            flow_seed = args.seeds[0]
            flow_checkpoint = Path(args.checkpoint_dir) / build_checkpoint_name("flow", phase_a_run_name(args.run_prefix, flow_seed))
            run_command(
                build_eval_command(
                    args,
                    python_bin,
                    output_csv=viz_main_dir / f"flow_seed{flow_seed}_viz.csv",
                    policy="flow",
                    model_path=flow_checkpoint,
                    run_name=f"{args.run_prefix}_flow_viz",
                    train_seed=flow_seed,
                    num_consensus_samples=args.default_consensus_samples,
                    viz_dir=viz_main_dir / "flow",
                    max_scenarios=args.viz_max_scenarios,
                    agent_counts=args.viz_agent_counts,
                ),
                args.dry_run,
            )


def stage_summarize(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    summary_dir = benchmark_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    main_rows = read_csv_rows(sorted((benchmark_dir / "evals" / "main").glob("*.csv")))
    if not main_rows:
        raise RuntimeError("No main evaluation CSVs found to summarize")
    summarize_main(main_rows, summary_dir)

    sweep_rows = read_csv_rows(sorted((benchmark_dir / "evals" / "sweeps").glob("*.csv")))
    summarize_consensus(sweep_rows, summary_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase A continuous MAPF benchmark workflow")
    parser.add_argument("--stages", nargs="+", default=["generate", "train", "eval", "summarize"])

    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="+", default=["empty-48-48"])
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[32, 64, 96, 128])
    parser.add_argument("--max-scenarios", type=int, default=5)

    parser.add_argument("--data-dir", default="data/continuous_phase_a")
    parser.add_argument("--checkpoint-dir", default="checkpoints/continuous_phase_a")
    parser.add_argument("--benchmark-dir", default="benchmarks/continuous_phase_a")
    parser.add_argument("--run-prefix", default="phaseA_empty48")

    parser.add_argument("--train-policies", nargs="+", default=["flow", "discrete"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--no-weighted-sampling", action="store_true")

    parser.add_argument("--expert-source", choices=["eecbs", "orca", "hybrid"], default="hybrid")
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--suboptimality", type=float, default=1.2)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--eecbs-repo", default=str(REPO_ROOT.parent / "EECBS-flow"))
    parser.add_argument("--eecbs-binary", default=None)

    parser.add_argument("--shield-type", choices=["orca", "heuristic-orca", "simple", "none"], default="orca")
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--default-consensus-samples", type=int, default=3)
    parser.add_argument("--enable-consensus-sweep", action="store_true")
    parser.add_argument("--consensus-sweep", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--max-eval-steps", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=0)

    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)

    parser.add_argument("--make-viz", action="store_true")
    parser.add_argument("--viz-agent-counts", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--viz-max-scenarios", type=int, default=1)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stages = set(args.stages)
    python_bin = sys.executable

    if "generate" in stages:
        stage_generate(args, python_bin)
    if "train" in stages:
        stage_train(args, python_bin)
    if "eval" in stages:
        stage_eval(args, python_bin)
    if "summarize" in stages:
        stage_summarize(args)


if __name__ == "__main__":
    main()
