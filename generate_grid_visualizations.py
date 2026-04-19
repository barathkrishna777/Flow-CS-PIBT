import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def resolve_path(path: str, base_dir: Path = REPO_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_by_name(root: Path, filename: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def resolve_existing_path(path: str, fallback_roots: Iterable[Path]) -> Path:
    resolved = resolve_path(path)
    if resolved.exists():
        return resolved
    for root in fallback_roots:
        match = find_by_name(resolve_path(str(root)), resolved.name)
        if match is not None:
            return match
    return resolved


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def is_successful_grid_row(row: Dict[str, str]) -> bool:
    if "mapName" not in row or "scenFile" not in row or "agentNum" not in row:
        return False
    success = parse_bool(row.get("success", ""))
    if not success:
        return False
    agent_num = parse_int(row, "agentNum")
    at_goal = parse_int(row, "num_agents_at_goal", agent_num)
    return at_goal >= agent_num


def select_difficult_successes(
    rows: Iterable[Dict[str, str]],
    top_k: int,
    min_agents: int,
    max_agents: Optional[int],
) -> List[Dict[str, str]]:
    candidates = []
    for row in rows:
        if not is_successful_grid_row(row):
            continue
        agent_num = parse_int(row, "agentNum")
        if agent_num < min_agents:
            continue
        if max_agents is not None and agent_num > max_agents:
            continue
        candidates.append(row)

    def difficulty_key(row: Dict[str, str]):
        return (
            parse_int(row, "agentNum"),
            parse_float(row, "total_cost_true"),
            parse_float(row, "runtime"),
        )

    return sorted(candidates, key=difficulty_key, reverse=True)[:top_k]


def row_difficulty_key(row: Dict[str, str]):
    return (
        parse_int(row, "agentNum"),
        parse_float(row, "total_cost_true"),
        parse_float(row, "runtime"),
    )


def select_successes_for_agent_counts(
    rows: Iterable[Dict[str, str]],
    agent_counts: List[int],
    distinct_maps: bool,
    selection: str,
    map_preferences: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    return [
        candidates[0]
        for candidates in select_candidate_lists_for_agent_counts(
            rows,
            agent_counts=agent_counts,
            distinct_maps=distinct_maps,
            selection=selection,
            map_preferences=map_preferences,
        )
    ]


def select_candidate_lists_for_agent_counts(
    rows: Iterable[Dict[str, str]],
    agent_counts: List[int],
    distinct_maps: bool,
    selection: str,
    map_preferences: Optional[List[str]] = None,
) -> List[List[Dict[str, str]]]:
    successful_rows = [row for row in rows if is_successful_grid_row(row)]
    selected: List[List[Dict[str, str]]] = []
    used_maps = set()
    missing_counts = []

    for index, agent_count in enumerate(agent_counts):
        map_preference = None
        if map_preferences and index < len(map_preferences):
            map_preference = map_preferences[index]
        candidates = [
            row
            for row in successful_rows
            if parse_int(row, "agentNum") == agent_count
            and (not distinct_maps or row["mapName"] not in used_maps)
            and map_matches_preference(row["mapName"], map_preference)
        ]
        if not candidates:
            candidates = [
                row
                for row in successful_rows
                if parse_int(row, "agentNum") >= agent_count
                and (not distinct_maps or row["mapName"] not in used_maps)
                and map_matches_preference(row["mapName"], map_preference)
            ]
        if not candidates:
            missing_counts.append(agent_count)
            continue
        if selection == "hardest":
            sort_key = lambda row: (
                abs(parse_int(row, "agentNum") - agent_count),
                -parse_int(row, "agentNum"),
                -parse_float(row, "total_cost_true"),
                -parse_float(row, "runtime"),
            )
        else:
            sort_key = lambda row: (
                abs(parse_int(row, "agentNum") - agent_count),
                parse_float(row, "runtime"),
                parse_float(row, "total_cost_true"),
            )
        candidate_list = []
        for row in sorted(candidates, key=sort_key):
            candidate = dict(row)
            source_agent_count = parse_int(candidate, "agentNum")
            candidate["sourceAgentNum"] = str(source_agent_count)
            candidate["agentNum"] = str(agent_count)
            candidate_list.append(candidate)
        selected.append(candidate_list)
        used_maps.add(candidate_list[0]["mapName"])

    if missing_counts:
        raise SystemExit(
            "Could not find successful grid-world rows for requested agent counts: "
            + ", ".join(str(count) for count in missing_counts)
        )
    return selected


def case_slug(row: Dict[str, str]) -> str:
    scen = Path(row["scenFile"]).stem
    return f"{row['mapName']}_{scen}_N{parse_int(row, 'agentNum')}_seed{parse_int(row, 'seed')}"


def describe_selected_case(row: Dict[str, str]) -> str:
    source_agents = row.get("sourceAgentNum")
    requested_agents = parse_int(row, "agentNum")
    source_text = ""
    if source_agents and parse_int(row, "sourceAgentNum") != requested_agents:
        source_text = f" from_successful_N{source_agents}"
    return (
        f"  {case_slug(row)}{source_text} "
        f"cost={row.get('total_cost_true')} runtime={row.get('runtime')}s"
    )


def map_matches_preference(map_name: str, preference: Optional[str]) -> bool:
    if not preference:
        return True
    normalized = map_name.lower()
    tokens = [
        token.strip().lower()
        for token in preference.replace("/", ",").split(",")
        if token.strip()
    ]
    return any(token in normalized for token in tokens)


def apply_showcase_defaults(args: argparse.Namespace) -> None:
    if not args.showcase:
        return
    if args.agent_counts is None:
        args.agent_counts = [50, 200, 1000]
    if args.map_preferences is None:
        args.map_preferences = ["berlin,paris", "den", "random"]
    args.distinct_maps = True
    args.agent_count_selection = "fastest"
    args.frame_stride = max(args.frame_stride, 12)
    args.trail_length = max(args.trail_length, 24)
    args.agent_size = max(args.agent_size, 28.0)
    args.agent_edge_width = max(args.agent_edge_width, 0.35)
    args.goal_size = max(args.goal_size, 44.0)
    args.goal_edge_width = max(args.goal_edge_width, 0.45)
    args.trail_width = max(args.trail_width, 1.8)
    args.figure_size = max(args.figure_size, 12.0)
    args.dpi = max(args.dpi, 180)
    args.frame_duration_ms = max(args.frame_duration_ms, 160)
    args.end_frame_duration_ms = max(args.end_frame_duration_ms, 2500)
    args.soft_style = True
    args.agent_edge_color = "#f7f5ee"
    args.goal_edge_color = "#f7f5ee"
    args.trail_alpha = max(args.trail_alpha, 0.5)
    args.agent_alpha = min(args.agent_alpha, 0.92)
    args.goal_alpha = min(args.goal_alpha, 0.9)
    args.background_color = "#fbfaf7"
    args.free_cell_color = "#fbfaf7"
    args.obstacle_cell_color = "#a3a8ad"
    args.map_interpolation = "bilinear"


def infer_bd_path(row: Dict[str, str], bd_dir: Path) -> Path:
    scen_name = Path(row["scenFile"]).stem
    filename = f"{scen_name}_bds.npz"
    direct = bd_dir / filename
    if direct.exists():
        return direct
    for candidate_dir in [
        REPO_ROOT / "data" / "bd_npzs" / "large_scale",
        REPO_ROOT / "data" / "constant_npzs",
        REPO_ROOT / "data" / "constant_npzs" / "bd_npzs",
        REPO_ROOT / "data" / "bd_npzs",
    ]:
        candidate = candidate_dir / filename
        if candidate.exists():
            return candidate
    match = find_by_name(REPO_ROOT / "data", filename)
    return match if match is not None else direct


def resolve_map_npz(path: str) -> Path:
    direct = resolve_path(path)
    if direct.exists():
        return direct
    fallback = first_existing(
        [
            REPO_ROOT / "data" / "all_maps.npz",
            REPO_ROOT / "data" / "constant_npzs" / "all_maps.npz",
            REPO_ROOT / "data" / "constant_npzs" / "all_maps.npz.npz",
        ]
    )
    return fallback if fallback is not None else direct


def resolve_scen_file(path: str) -> Path:
    direct = resolve_path(path)
    if direct.exists():
        return direct
    filename = direct.name
    fallback = first_existing(
        [
            REPO_ROOT / "data" / "scen-random" / filename,
            REPO_ROOT / "data" / "mapf-scen-random" / filename,
        ]
    )
    return fallback if fallback is not None else direct


def resolve_model_path(row: Dict[str, str], override: Optional[str]) -> Path:
    if override:
        return resolve_existing_path(
            override,
            [
                REPO_ROOT,
                REPO_ROOT / "checkpoints",
                REPO_ROOT / "data",
            ],
        )
    model_path = row.get("modelPath", "")
    return resolve_existing_path(
        model_path,
        [
            REPO_ROOT,
            REPO_ROOT / "checkpoints",
            REPO_ROOT / "data",
        ],
    )


def choose_gpu_value(row: Dict[str, str], mode: str) -> str:
    if mode == "true":
        return "True"
    if mode == "false":
        return "False"
    return "True" if parse_bool(row.get("useGPU", "False")) else "False"


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def verify_rerun_success(metrics_file: Path, agent_num: int) -> None:
    rows = read_rows(metrics_file)
    if not rows:
        raise RuntimeError(f"No simulator metrics were written to {metrics_file}")
    row = rows[-1]
    if not parse_bool(row.get("success", "")) or parse_int(row, "num_agents_at_goal") < agent_num:
        raise RuntimeError(
            "Rerun did not solve the selected case, so no GIF was rendered: "
            f"{metrics_file} success={row.get('success')} "
            f"num_agents_at_goal={row.get('num_agents_at_goal')}/{agent_num}"
        )


def rerun_case(
    row: Dict[str, str],
    args: argparse.Namespace,
    paths_file: Path,
    metrics_file: Path,
) -> None:
    map_npz = resolve_map_npz(args.map_npz)
    scen_file = resolve_scen_file(row["scenFile"])
    model_path = resolve_model_path(row, args.model_path)
    bd_path = infer_bd_path(row, resolve_path(args.bd_dir))

    missing = [
        str(path)
        for path in [map_npz, scen_file, bd_path, model_path]
        if not path.exists()
    ]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "Cannot rerun case because these inputs are missing:\n  "
            + "\n  ".join(missing)
        )
    if missing:
        print(
            "Dry run: these inputs are not present in this workspace:\n  "
            + "\n  ".join(missing),
            flush=True,
        )

    paths_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "main_pys.simulator",
        f"--mapNpzFile={map_npz}",
        f"--mapName={row['mapName']}",
        f"--scenFile={scen_file}",
        f"--agentNum={parse_int(row, 'agentNum')}",
        f"--bdNpzFile={bd_path}",
        f"--modelPath={model_path}",
        f"--useGPU={choose_gpu_value(row, args.use_gpu)}",
        f"--maxSteps={row.get('maxSteps') or args.max_steps}",
        f"--seed={parse_int(row, 'seed')}",
        f"--shieldType={row.get('shieldType') or args.shield_type}",
        f"--lacamLookahead={parse_int(row, 'lacamLookahead', args.lacam_lookahead)}",
        f"--timeLimit={args.time_limit}",
        f"--outputCSVFile={metrics_file}",
        f"--outputPathsFile={paths_file}",
        f"--policyType={args.policy_type}",
        f"--numIntegrationSteps={args.num_integration_steps}",
        f"--numConsensusSamples={args.num_consensus_samples}",
        f"--tau={args.tau}",
        f"--waitThreshold={args.wait_threshold}",
        f"--hiddenDim={args.hidden_dim}",
        f"--numLayers={args.num_layers}",
    ]
    run_command(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        verify_rerun_success(metrics_file, parse_int(row, "agentNum"))


def render_case(
    row: Dict[str, str],
    args: argparse.Namespace,
    paths_file: Path,
    gif_file: Path,
) -> None:
    scen_file = resolve_scen_file(row["scenFile"])
    frame_dir = gif_file.parent / f"{gif_file.stem}_frames"
    cmd = [
        sys.executable,
        "-m",
        "main_pys.visualize_path",
        row["mapName"],
        str(paths_file),
        f"--scenName={scen_file.name}",
        f"--mapFolder={resolve_path(args.map_dir)}",
        f"--sceneFile={scen_file.parent}",
        f"--outputGif={gif_file}",
        f"--tmpFolderToSaveImages={frame_dir}",
        f"--frameStride={args.frame_stride}",
        f"--trailLength={args.trail_length}",
        f"--agentSize={args.agent_size}",
        f"--goalSize={args.goal_size}",
        f"--trailWidth={args.trail_width}",
        f"--agentEdgeWidth={args.agent_edge_width}",
        f"--goalEdgeWidth={args.goal_edge_width}",
        f"--agentEdgeColor={args.agent_edge_color}",
        f"--goalEdgeColor={args.goal_edge_color}",
        f"--trailAlpha={args.trail_alpha}",
        f"--agentAlpha={args.agent_alpha}",
        f"--goalAlpha={args.goal_alpha}",
        f"--figureSize={args.figure_size}",
        f"--dpi={args.dpi}",
        f"--frameDurationMs={args.frame_duration_ms}",
        f"--endFrameDurationMs={args.end_frame_duration_ms}",
        f"--backgroundColor={args.background_color}",
        f"--freeCellColor={args.free_cell_color}",
        f"--obstacleCellColor={args.obstacle_cell_color}",
        f"--mapInterpolation={args.map_interpolation}",
    ]
    if args.soft_style:
        cmd.append("--softStyle")
    run_command(cmd, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pick hard successful grid-world simulator cases from a CSV, rerun "
            "them to save paths, and render GIF visualizations."
        )
    )
    parser.add_argument("--eval-csv", required=True, help="Simulator-format CSV to mine for successful cases")
    parser.add_argument("--output-dir", default="visualizations/grid_successes")
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Convenience preset for 3 simple GIFs: 50, 200, 1000 agents on Paris/Berlin, den, and random maps.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--agent-counts",
        nargs="+",
        type=int,
        default=None,
        help="Select exactly one successful case for each requested agent count, in this order.",
    )
    parser.add_argument(
        "--distinct-maps",
        action="store_true",
        help="When used with --agent-counts, do not select the same map twice.",
    )
    parser.add_argument(
        "--agent-count-selection",
        choices=["fastest", "hardest"],
        default="fastest",
        help="How to choose cases for --agent-counts. Defaults to fastest because this mode is for quick GIF generation.",
    )
    parser.add_argument(
        "--map-preferences",
        nargs="+",
        default=None,
        help=(
            "Optional map-name preferences, one per --agent-counts entry. "
            "Use comma-separated alternatives like 'berlin,paris'."
        ),
    )
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--map-npz", default="data/all_maps.npz")
    parser.add_argument("--map-dir", default="data/mapf-map")
    parser.add_argument("--bd-dir", default="data/bd_npzs/large_scale")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the checkpoint path from the CSV. Useful when the CSV was produced on another machine.",
    )
    parser.add_argument("--max-steps", default="3x")
    parser.add_argument("--shield-type", default="CS-PIBT")
    parser.add_argument("--lacam-lookahead", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--use-gpu", choices=["csv", "true", "false"], default="csv")
    parser.add_argument("--policy-type", choices=["flow", "classifier", "flow_action_head", "local_classifier"], default="flow")
    parser.add_argument("--num-integration-steps", type=int, default=5)
    parser.add_argument("--num-consensus-samples", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--wait-threshold", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--agent-size", type=float, default=10.0)
    parser.add_argument("--agent-edge-width", type=float, default=0.2)
    parser.add_argument("--goal-size", type=float, default=24.0)
    parser.add_argument("--goal-edge-width", type=float, default=0.25)
    parser.add_argument("--trail-width", type=float, default=0.7)
    parser.add_argument("--agent-edge-color", default="black")
    parser.add_argument("--goal-edge-color", default="black")
    parser.add_argument("--trail-alpha", type=float, default=0.35)
    parser.add_argument("--agent-alpha", type=float, default=1.0)
    parser.add_argument("--goal-alpha", type=float, default=0.8)
    parser.add_argument("--figure-size", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--frame-duration-ms", type=int, default=70)
    parser.add_argument("--end-frame-duration-ms", type=int, default=1600)
    parser.add_argument("--soft-style", action="store_true")
    parser.add_argument("--background-color", default="#fbfaf7")
    parser.add_argument("--free-cell-color", default="#fbfaf7")
    parser.add_argument("--obstacle-cell-color", default="#9fa4aa")
    parser.add_argument("--map-interpolation", default="nearest")
    parser.add_argument("--reuse-paths", action="store_true", help="Skip simulator reruns when a paths .npy already exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_showcase_defaults(args)

    rows = read_rows(resolve_path(args.eval_csv))
    if args.agent_counts:
        candidate_lists = select_candidate_lists_for_agent_counts(
            rows,
            agent_counts=args.agent_counts,
            distinct_maps=args.distinct_maps,
            selection=args.agent_count_selection,
            map_preferences=args.map_preferences,
        )
        selected = [candidates[0] for candidates in candidate_lists]
    else:
        selected = select_difficult_successes(rows, args.top_k, args.min_agents, args.max_agents)
        candidate_lists = [[row] for row in selected]
    if not selected:
        raise SystemExit("No successful grid-world rows matched the requested filters.")

    output_dir = resolve_path(args.output_dir)
    paths_dir = output_dir / "paths"
    gifs_dir = output_dir / "gifs"
    metrics_dir = output_dir / "metrics"
    print(f"Selected {len(selected)} successful cases:", flush=True)
    for row in selected:
        print(describe_selected_case(row), flush=True)

    for candidates in candidate_lists:
        last_error = None
        for attempt_index, row in enumerate(candidates, start=1):
            slug = case_slug(row)
            paths_file = paths_dir / f"{slug}.npy"
            gif_file = gifs_dir / f"{slug}.gif"
            metrics_file = metrics_dir / f"{slug}.csv"
            if attempt_index > 1:
                print(f"Retrying with alternate successful CSV case: {describe_selected_case(row)}", flush=True)
            try:
                if not args.reuse_paths or not paths_file.exists():
                    rerun_case(row, args, paths_file, metrics_file)
                else:
                    print(f"Reusing {paths_file}", flush=True)
                    if not args.dry_run:
                        verify_rerun_success(metrics_file, parse_int(row, "agentNum"))
                render_case(row, args, paths_file, gif_file)
                verb = "Would write" if args.dry_run else "Wrote"
                print(f"{verb} {gif_file}", flush=True)
                last_error = None
                break
            except RuntimeError as error:
                last_error = error
                print(f"Case failed verification: {error}", flush=True)
                continue
        if last_error is not None:
            raise last_error


if __name__ == "__main__":
    main()
