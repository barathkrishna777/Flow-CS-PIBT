import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def scenario_range_count(start: Optional[int], end: Optional[int]) -> Optional[int]:
    if start is None or end is None:
        return None
    return max(end - start + 1, 0)


def apply_funnel_stage_defaults(args: argparse.Namespace) -> None:
    if args.funnel_stage == "custom":
        return
    presets = {
        "screen": {
            "epochs": 10,
            "seeds": [0],
            "train_scenario_start": 1,
            "train_scenario_end": 10,
            "val_scenario_start": 11,
            "val_scenario_end": 15,
            "test_scenario_start": 11,
            "test_scenario_end": 15,
        },
        "medium": {
            "epochs": 20,
            "seeds": [0],
            "train_scenario_start": 1,
            "train_scenario_end": 50,
            "val_scenario_start": 51,
            "val_scenario_end": 60,
            "test_scenario_start": 51,
            "test_scenario_end": 60,
        },
        "official": {
            "epochs": 30,
            "seeds": [0, 1, 2],
            "train_scenario_start": 1,
            "train_scenario_end": 50,
            "val_scenario_start": 51,
            "val_scenario_end": 60,
            "test_scenario_start": 61,
            "test_scenario_end": 80,
        },
    }
    for key, value in presets[args.funnel_stage].items():
        setattr(args, key, value)


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
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["policy"], row.get("flow_aggregation", ""), int(row["agents"]))
        for metric in metrics:
            grouped[key][metric].append(float(row[metric]))

    summary_rows = []
    for (policy, flow_aggregation, agents), metric_values in sorted(grouped.items()):
        summary_row: Dict[str, object] = {"policy": policy, "flow_aggregation": flow_aggregation, "agents": agents}
        for metric in metrics:
            avg, std = mean_std(metric_values.get(metric, []))
            summary_row[f"{metric}_mean"] = avg
            summary_row[f"{metric}_std"] = std
        summary_rows.append(summary_row)

    summary_fields = ["policy", "flow_aggregation", "agents"] + [f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "std")]
    write_csv(output_dir / "main_summary.csv", summary_fields, summary_rows)

    policies = sorted({(row["policy"], row["flow_aggregation"]) for row in summary_rows})
    agent_counts = sorted({int(row["agents"]) for row in summary_rows})
    main_table_rows = []
    for policy, flow_aggregation in policies:
        label = policy if not flow_aggregation else f"{policy}[{flow_aggregation}]"
        row: Dict[str, object] = {"policy": label}
        for agents in agent_counts:
            match = next(
                (
                    entry
                    for entry in summary_rows
                    if entry["policy"] == policy
                    and entry["flow_aggregation"] == flow_aggregation
                    and entry["agents"] == agents
                ),
                None,
            )
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
        for policy, flow_aggregation in policies:
            policy_rows = sorted(
                [
                    entry
                    for entry in summary_rows
                    if entry["policy"] == policy and entry["flow_aggregation"] == flow_aggregation
                ],
                key=lambda entry: int(entry["agents"]),
            )
            xs = [int(entry["agents"]) for entry in policy_rows]
            ys = [float(entry[f"{metric}_mean"]) for entry in policy_rows]
            yerr = [float(entry[f"{metric}_std"]) for entry in policy_rows]
            label = policy if not flow_aggregation else f"{policy}[{flow_aggregation}]"
            plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=2, label=label)
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

    grouped: Dict[Tuple[int, str, int], List[float]] = defaultdict(list)
    runtime_grouped: Dict[Tuple[int, str, int], List[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["num_consensus_samples"]), row.get("flow_aggregation", ""), int(row["agents"]))
        grouped[key].append(float(row["agent_fraction_at_goal"]))
        runtime_grouped[key].append(float(row["runtime"]))

    summary_rows = []
    consensus_values = sorted({key[0] for key in grouped})
    aggregations = sorted({key[1] for key in grouped})
    agent_counts = sorted({key[2] for key in grouped})
    for consensus, aggregation, agents in sorted(grouped):
        goal_mean, goal_std = mean_std(grouped[(consensus, aggregation, agents)])
        runtime_mean, runtime_std = mean_std(runtime_grouped[(consensus, aggregation, agents)])
        summary_rows.append(
            {
                "num_consensus_samples": consensus,
                "flow_aggregation": aggregation,
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
            "flow_aggregation",
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
        for aggregation in aggregations:
            for consensus in consensus_values:
                subset = sorted(
                    [
                        row
                        for row in summary_rows
                        if int(row["num_consensus_samples"]) == consensus and row["flow_aggregation"] == aggregation
                    ],
                    key=lambda row: int(row["agents"]),
                )
                if not subset:
                    continue
                xs = [int(row["agents"]) for row in subset]
                ys = [float(row[f"{metric}_mean"]) for row in subset]
                yerr = [float(row[f"{metric}_std"]) for row in subset]
                label = f"c={consensus}" if not aggregation else f"c={consensus}[{aggregation}]"
                plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=2, label=label)
        plt.xlabel("Agents")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()


def summarize_dataset(data_dir: Path, output_dir: Path) -> None:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        return

    rollout_rows: List[Dict[str, object]] = []
    source_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if "expert_source_used" in data:
                expert_source = str(data["expert_source_used"].item())
            else:
                expert_source = str(data["expert_source"].item())
            agent_count = int(data["agent_count"].item()) if "agent_count" in data else int(data["positions"].shape[1])
            scenario_id = int(data["scenario_id"].item()) if "scenario_id" in data else -1
            rollout_rows.append(
                {
                    "file": path.name,
                    "expert_source": expert_source,
                    "agent_count": agent_count,
                    "scenario_id": scenario_id,
                    "rollout_length": int(data["rollout_length"].item()) if "rollout_length" in data else int(data["positions"].shape[0] - 1),
                    "fraction_moving": float(data["fraction_moving"].item()) if "fraction_moving" in data else 0.0,
                    "mean_nearest_neighbor_distance": (
                        float(data["mean_nearest_neighbor_distance"].item()) if "mean_nearest_neighbor_distance" in data else 0.0
                    ),
                }
            )
            source_counts[(expert_source, agent_count)] += 1

    write_csv(
        output_dir / "dataset_rollout_summary.csv",
        ["file", "expert_source", "agent_count", "scenario_id", "rollout_length", "fraction_moving", "mean_nearest_neighbor_distance"],
        rollout_rows,
    )
    write_csv(
        output_dir / "dataset_source_summary.csv",
        ["expert_source", "agent_count", "num_rollouts"],
        [
            {"expert_source": source, "agent_count": agents, "num_rollouts": count}
            for (source, agents), count in sorted(source_counts.items())
        ],
    )


def phase_a_run_name(prefix: str, seed: int) -> str:
    return f"{prefix}_s{seed}"


def stage_generate(args: argparse.Namespace, python_bin: str) -> None:
    scenario_starts = [
        value
        for value in [args.train_scenario_start, args.val_scenario_start, args.test_scenario_start]
        if value is not None
    ]
    scenario_ends = [
        value
        for value in [args.train_scenario_end, args.val_scenario_end, args.test_scenario_end]
        if value is not None
    ]
    effective_max_scenarios = args.max_scenarios
    if scenario_starts and scenario_ends:
        effective_max_scenarios = max(effective_max_scenarios, max(scenario_ends) - min(scenario_starts) + 1)
    cmd = [
        python_bin,
        "-m",
        "scripts.generate_continuous_data",
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
        str(effective_max_scenarios),
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
    if scenario_starts:
        cmd.extend(["--scenario-start", str(min(scenario_starts))])
    if scenario_ends:
        cmd.extend(["--scenario-end", str(max(scenario_ends))])
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
            if args.train_scenario_start is not None:
                cmd.extend(["--train-scenario-start", str(args.train_scenario_start)])
            if args.train_scenario_end is not None:
                cmd.extend(["--train-scenario-end", str(args.train_scenario_end)])
            if args.val_scenario_start is not None:
                cmd.extend(["--val-scenario-start", str(args.val_scenario_start)])
            if args.val_scenario_end is not None:
                cmd.extend(["--val-scenario-end", str(args.val_scenario_end)])
            if args.expert_source_filter:
                cmd.extend(["--expert-sources", *args.expert_source_filter])
            if args.oversample_difficult:
                cmd.append("--oversample-difficult")
            if args.shield_aware_loss:
                cmd.append("--shield-aware-loss")
            cmd.extend(
                [
                    "--dt",
                    str(args.dt),
                    "--agent-radius",
                    str(args.agent_radius),
                    "--train-shield-type",
                    args.train_shield_type,
                    "--flow-loss-weight",
                    str(args.flow_loss_weight),
                    "--shield-loss-weight",
                    str(args.shield_loss_weight),
                    "--action-loss-weight",
                    str(args.action_loss_weight),
                ]
            )
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
    flow_aggregation: str,
    viz_dir: Optional[Path] = None,
    max_scenarios: Optional[int] = None,
    agent_counts: Optional[Sequence[int]] = None,
) -> List[str]:
    effective_max_scenarios = max_scenarios or args.max_scenarios
    eval_range_count = scenario_range_count(args.test_scenario_start, args.test_scenario_end)
    if eval_range_count is not None:
        effective_max_scenarios = max(effective_max_scenarios, eval_range_count)
    cmd = [
        python_bin,
        "-m",
        "scripts.eval_continuous",
        "--map-dir",
        args.map_dir,
        "--scen-dir",
        args.scen_dir,
        "--maps",
        *args.maps,
        "--agent-counts",
        *[str(agent_count) for agent_count in (agent_counts or args.agent_counts)],
        "--max-scenarios",
        str(effective_max_scenarios),
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
        "--flow-aggregation",
        flow_aggregation,
        "--tau",
        str(args.tau),
        "--k",
        str(args.k),
        "--m",
        str(args.m),
        "--max-steps",
        str(args.max_eval_steps),
        "--log-interval",
        str(args.log_interval),
        "--dt",
        str(args.dt),
        "--max-speed",
        str(args.max_speed),
        "--agent-radius",
        str(args.agent_radius),
        "--goal-tolerance",
        str(args.goal_tolerance),
    ]
    if args.picbf_communication_radius is not None:
        cmd.extend(["--picbf-communication-radius", str(args.picbf_communication_radius)])
    if args.test_scenario_start is not None:
        cmd.extend(["--scenario-start", str(args.test_scenario_start)])
    if args.test_scenario_end is not None:
        cmd.extend(["--scenario-end", str(args.test_scenario_end)])
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
    if args.enable_consensus_sweep or args.enable_aggregation_sweep:
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
                flow_aggregation="mean",
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
                        flow_aggregation=args.default_flow_aggregation if policy == "flow" else "mean",
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
                            flow_aggregation=args.default_flow_aggregation,
                        ),
                        args.dry_run,
                    )
            if policy == "flow" and args.enable_aggregation_sweep:
                for aggregation in args.aggregation_sweep:
                    if aggregation == args.default_flow_aggregation:
                        continue
                    sweep_csv = eval_sweep_dir / f"flow_{aggregation}_seed{seed}.csv"
                    if args.skip_existing and sweep_csv.exists():
                        print(f"Skipping existing aggregation eval: {sweep_csv}")
                        continue
                    run_command(
                        build_eval_command(
                            args,
                            python_bin,
                            output_csv=sweep_csv,
                            policy="flow",
                            model_path=checkpoint_path,
                            run_name=f"{args.run_prefix}_flow_aggregation_sweep",
                            train_seed=seed,
                            num_consensus_samples=args.default_consensus_samples,
                            flow_aggregation=aggregation,
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
                flow_aggregation="mean",
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
                    flow_aggregation=args.default_flow_aggregation,
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
    summarize_dataset(Path(args.data_dir), summary_dir)

    main_rows = read_csv_rows(sorted((benchmark_dir / "evals" / "main").glob("*.csv")))
    if not main_rows:
        raise RuntimeError("No main evaluation CSVs found to summarize")
    summarize_main(main_rows, summary_dir)

    sweep_rows = read_csv_rows(sorted((benchmark_dir / "evals" / "sweeps").glob("*.csv")))
    summarize_consensus(sweep_rows, summary_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase A continuous MAPF benchmark workflow")
    parser.add_argument("--stages", nargs="+", default=["generate", "train", "eval", "summarize"])
    parser.add_argument("--funnel-stage", choices=["screen", "medium", "official", "custom"], default="custom")

    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="+", default=["empty-48-48"])
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[32, 64, 96, 128])
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--train-scenario-start", type=int, default=None)
    parser.add_argument("--train-scenario-end", type=int, default=None)
    parser.add_argument("--val-scenario-start", type=int, default=None)
    parser.add_argument("--val-scenario-end", type=int, default=None)
    parser.add_argument("--test-scenario-start", type=int, default=None)
    parser.add_argument("--test-scenario-end", type=int, default=None)

    parser.add_argument("--data-dir", default="data/continuous_main/raw")
    parser.add_argument("--checkpoint-dir", default="checkpoints/continuous_main")
    parser.add_argument("--benchmark-dir", default="benchmarks/continuous_main")
    parser.add_argument("--run-prefix", default="continuous_main")

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
    parser.add_argument("--oversample-difficult", action="store_true")
    parser.add_argument("--expert-source-filter", nargs="+", default=None)
    parser.add_argument("--shield-aware-loss", action="store_true")
    parser.add_argument("--train-shield-type", choices=["orca", "heuristic-orca"], default="orca")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--shield-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.1)

    parser.add_argument("--expert-source", choices=["eecbs", "orca", "hybrid"], default="hybrid")
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--suboptimality", type=float, default=1.2)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--eecbs-repo", default=str(REPO_ROOT.parent / "EECBS-flow"))
    parser.add_argument("--eecbs-binary", default=None)

    parser.add_argument(
        "--shield-type",
        choices=["orca", "heuristic-orca", "po-orca", "epibt", "picbf-cs", "simple", "none"],
        default="orca",
    )
    parser.add_argument(
        "--picbf-communication-radius",
        type=float,
        default=None,
        help=(
            "Communication radius forwarded to scripts/eval_continuous.py for "
            "--shield-type picbf-cs. Defaults to scripts/eval_continuous.py auto-compute."
        ),
    )
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--default-consensus-samples", type=int, default=3)
    parser.add_argument("--default-flow-aggregation", choices=["mean", "medoid", "best"], default="mean")
    parser.add_argument("--enable-consensus-sweep", action="store_true")
    parser.add_argument("--consensus-sweep", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--enable-aggregation-sweep", action="store_true")
    parser.add_argument("--aggregation-sweep", nargs="+", choices=["mean", "medoid", "best"], default=["mean", "medoid", "best"])
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--max-eval-steps", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=10)
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
    if args.picbf_communication_radius is not None and args.picbf_communication_radius <= 0.0:
        parser.error("--picbf-communication-radius must be positive")
    apply_funnel_stage_defaults(args)
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
