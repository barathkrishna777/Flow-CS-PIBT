"""Inspect tracker-preferred labels in a continuous-MAPF rollout npz.

Usage:
    python scripts/inspect_tracker_labels.py <path.npz> [--agents 0 1 2]

Prints dataset-wide diagnostics that let us decide whether
`tracker_preferred_velocities` actually looks like a goal-directed EECBS
tracker signal, or whether it has been mis-stored / mis-computed.

No training/eval — pure numpy, CPU, finishes in seconds.
"""

import argparse
import numpy as np


def _mean_cos_with_goal(velocities_t, positions_t, goals, wait_threshold):
    """Mean cosine of velocities[t, i] with (goal[i] - positions[t, i]).

    Only counts agents where both the velocity and the goal vector are
    non-trivial (velocity above wait_threshold, goal distance above 1e-3).
    Returns (mean_cosine, fraction_of_agent_steps_counted).
    """
    T = velocities_t.shape[0]
    values = []
    counted = 0
    total = 0
    for t in range(T):
        v = velocities_t[t]
        p = positions_t[t]
        gd = goals - p
        vn = np.linalg.norm(v, axis=1)
        gn = np.linalg.norm(gd, axis=1)
        mask = (vn > wait_threshold) & (gn > 1e-3)
        total += v.shape[0]
        counted += int(mask.sum())
        if mask.any():
            cos = np.sum(
                (v[mask] / vn[mask, None]) * (gd[mask] / gn[mask, None]),
                axis=1,
            )
            values.extend(cos.tolist())
    mean = float(np.mean(values)) if values else 0.0
    return mean, (counted / max(total, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("--agents", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument("--stall-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    keys = list(data.files)
    print(f"file: {args.npz_path}")
    print(f"keys: {sorted(keys)}")

    positions = data["positions"].astype(np.float32)        # (T+1, N, 2)
    velocities = data["velocities"].astype(np.float32)      # (T, N, 2)
    goals = data["goals"].astype(np.float32)                # (N, 2)
    preferred = data.get("tracker_preferred_velocities")    # (T, N, 2)
    waypoint_idx = data.get("tracker_waypoint_indices")     # (T, N)
    waypoint_counts = data.get("tracker_waypoint_counts")   # (N,)

    print(
        f"\nshapes: positions={positions.shape} velocities={velocities.shape} "
        f"goals={goals.shape}"
    )
    if preferred is None:
        print("!!! tracker_preferred_velocities MISSING — was this rollout "
              "generated with --expert-source eecbs-guided-orca?")
        return
    print(
        f"        tracker_preferred_velocities={preferred.shape} "
        f"waypoint_indices={None if waypoint_idx is None else waypoint_idx.shape} "
        f"waypoint_counts={None if waypoint_counts is None else waypoint_counts.shape}"
    )
    expert_source = data.get("expert_source_used", data.get("expert_source", np.asarray("?")))
    print(f"expert_source_used: {expert_source}")

    T = velocities.shape[0]
    N = velocities.shape[1]
    pos_t = positions[:T]  # aligned with velocities/preferred indexing

    # --- Norms / stall stats ---
    pref_norms = np.linalg.norm(preferred, axis=2)   # (T, N)
    exec_norms = np.linalg.norm(velocities, axis=2)  # (T, N)
    print("\n== speed stats (agent-step mean over T*N) ==")
    print(f"  preferred  mean_norm = {pref_norms.mean():.4f}  "
          f"median = {np.median(pref_norms):.4f}  max = {pref_norms.max():.4f}")
    print(f"  executed   mean_norm = {exec_norms.mean():.4f}  "
          f"median = {np.median(exec_norms):.4f}  max = {exec_norms.max():.4f}")
    pref_zero = (pref_norms <= args.stall_threshold).mean()
    exec_zero = (exec_norms <= args.stall_threshold).mean()
    print(f"  preferred  stall_fraction(|v|<={args.stall_threshold}) = {pref_zero:.4f}")
    print(f"  executed   stall_fraction(|v|<={args.stall_threshold}) = {exec_zero:.4f}")

    # --- Cosine with goal direction ---
    pref_cos, pref_frac = _mean_cos_with_goal(preferred, pos_t, goals, args.wait_threshold)
    exec_cos, exec_frac = _mean_cos_with_goal(velocities, pos_t, goals, args.wait_threshold)
    print("\n== goal-direction cosine (over moving agent-steps) ==")
    print(f"  preferred  mean_cos_with_goal = {pref_cos:.4f}  "
          f"frac_counted = {pref_frac:.4f}")
    print(f"  executed   mean_cos_with_goal = {exec_cos:.4f}  "
          f"frac_counted = {exec_frac:.4f}")

    # Compare preferred vs executed directly: do they look the same array?
    same = np.allclose(preferred, velocities, atol=1e-5)
    diff_mean = float(np.mean(np.linalg.norm(preferred - velocities, axis=2)))
    print(f"\n== preferred vs executed arrays ==")
    print(f"  allclose(preferred, velocities) = {same}")
    print(f"  mean_step_diff_norm = {diff_mean:.4f}  "
          f"(if ~0, preferred was mis-stored as executed)")

    # --- Per-step mean cosines over early / middle / late slices ---
    def slice_cos(v, label):
        cos, frac = _mean_cos_with_goal(v, pos_t, goals, args.wait_threshold)
        print(f"    {label}: cos={cos:.4f} frac={frac:.4f}")

    print("\n== preferred cosine by rollout quarter ==")
    q = max(1, T // 4)
    slice_cos(preferred[:q], "t in [0, T/4)   ")
    slice_cos(preferred[q:2 * q], "t in [T/4, T/2) ")
    slice_cos(preferred[2 * q:3 * q], "t in [T/2, 3T/4)")
    slice_cos(preferred[3 * q:], "t in [3T/4, T)  ")

    # --- Per-agent sanity print ---
    print("\n== per-agent sanity (first 3 steps) ==")
    for a in args.agents:
        if a >= N:
            continue
        start = positions[0, a]
        goal = goals[a]
        goal_dir = goal - start
        goal_dir_n = goal_dir / max(np.linalg.norm(goal_dir), 1e-6)
        print(f"agent {a}: start={start.tolist()} goal={goal.tolist()} "
              f"|goal-start|={float(np.linalg.norm(goal_dir)):.2f} "
              f"route_len={None if waypoint_counts is None else int(waypoint_counts[a])}")
        for t in range(min(3, T)):
            pv = preferred[t, a]
            ev = velocities[t, a]
            p = positions[t, a]
            gd = goals[a] - p
            gn = max(np.linalg.norm(gd), 1e-6)
            pvn = max(np.linalg.norm(pv), 1e-6)
            evn = max(np.linalg.norm(ev), 1e-6)
            cos_pref = float((pv @ gd) / (pvn * gn))
            cos_exec = float((ev @ gd) / (evn * gn))
            wi = None if waypoint_idx is None else int(waypoint_idx[t, a])
            print(
                f"  t={t} pos={p.tolist()} "
                f"pref=({pv[0]:+.3f},{pv[1]:+.3f}) |pref|={pvn:.3f} "
                f"cos(pref,goal)={cos_pref:+.3f} "
                f"exec=({ev[0]:+.3f},{ev[1]:+.3f}) |exec|={evn:.3f} "
                f"cos(exec,goal)={cos_exec:+.3f} waypoint_idx={wi}"
            )

    # --- Distribution of per-agent final cosine vs direct start->goal ---
    # If EECBS routes are sensible, average preferred direction over the
    # moving portion of the rollout should roughly point toward the goal.
    per_agent_cos = []
    for a in range(N):
        total = np.zeros(2, dtype=np.float64)
        count = 0
        for t in range(T):
            pv = preferred[t, a]
            if np.linalg.norm(pv) >= args.wait_threshold:
                total += pv
                count += 1
        if count == 0:
            continue
        gd = goals[a] - positions[0, a]
        gn = np.linalg.norm(gd)
        tn = np.linalg.norm(total)
        if gn > 1e-3 and tn > 1e-6:
            per_agent_cos.append(float(np.dot(total, gd) / (tn * gn)))
    if per_agent_cos:
        arr = np.asarray(per_agent_cos)
        print(f"\n== per-agent mean-preferred-direction vs start->goal cosine ==")
        print(f"  N_agents_with_motion = {len(arr)}  "
              f"mean = {arr.mean():.4f}  "
              f"median = {float(np.median(arr)):.4f}  "
              f"frac>0.5 = {float((arr > 0.5).mean()):.4f}  "
              f"frac<0 = {float((arr < 0).mean()):.4f}")


if __name__ == "__main__":
    main()
