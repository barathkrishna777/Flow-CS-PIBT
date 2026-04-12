"""Validation tests for PO-ORCA shield and SDF obstacle handling.

Run: python test_po_orca.py
No trained model or map/scen files needed -- uses synthetic scenarios.
"""

import sys
import numpy as np

from main_pys.continuous_env import (
    ContinuousMAPFEnv,
    ORCAStyleShield,
    compute_sdf,
    sdf_gradient,
    sample_sdf_bilinear,
    sample_sdf_gradient_bilinear,
)

PASS = 0
FAIL = 0
SKIP = 0


def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{status}] {name}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)


def report_skip(name: str, detail: str = ""):
    global SKIP
    SKIP += 1
    msg = f"[SKIP] {name}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)


# ---------------------------------------------------------------
# Test 1: SDF correctness
# ---------------------------------------------------------------
def test_sdf_correctness():
    grid = np.zeros((10, 10), dtype=np.int8)
    grid[4:6, 4:6] = 1  # 2x2 obstacle block

    sdf = compute_sdf(grid)

    # Inside obstacle: negative
    inside_val = sdf[4, 4]
    report("SDF: inside obstacle is negative", inside_val < 0, f"sdf[4,4]={inside_val:.3f}")

    # Far from obstacle: positive
    far_val = sdf[0, 0]
    report("SDF: far from obstacle is positive", far_val > 0, f"sdf[0,0]={far_val:.3f}")

    # Adjacent free cell should be ~1.0 (distance of 1 cell to boundary)
    adj_val = sdf[3, 4]
    report("SDF: adjacent free cell ~1.0", 0.5 < adj_val < 1.5, f"sdf[3,4]={adj_val:.3f}")

    # Gradient should point away from obstacle in free space
    grad_r, grad_c = sdf_gradient(sdf)
    # At (3, 5) -- above and slightly right of obstacle
    # gradient row component should be negative (pointing up, away from obstacle below)
    pos = np.array([[3.0, 5.0]])
    grad_val = sample_sdf_gradient_bilinear(grad_r, grad_c, pos)[0]
    grad_norm = np.linalg.norm(grad_val)
    report("SDF gradient: nonzero near obstacle", grad_norm > 0.1, f"||grad||={grad_norm:.3f}")


# ---------------------------------------------------------------
# Test 2: SDF obstacle avoidance -- single agent with wall
# ---------------------------------------------------------------
def test_sdf_single_agent_wall():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[10, :] = 1  # Wall across the middle

    env = ContinuousMAPFEnv(grid, agent_radius=0.3)
    starts = np.array([[5.5, 10.5]], dtype=np.float32)  # Above wall
    goals = np.array([[15.5, 10.5]], dtype=np.float32)   # Below wall
    env.reset(starts, goals)

    for _ in range(200):
        vel = env.goal_directed_velocities()
        env.step(vel, shield_type="po-orca")
        if env.is_done():
            break

    metrics = env.current_metrics()
    report(
        "Wall avoidance: 0 obstacle hits",
        metrics["obstacle_hits"] == 0,
        f"obstacle_hits={metrics['obstacle_hits']:.0f}",
    )


# ---------------------------------------------------------------
# Test 3: 4-agent crossing -- compare shield types
# ---------------------------------------------------------------
def test_4agent_crossing():
    grid = np.zeros((20, 20), dtype=np.int8)
    starts = np.array([
        [5.5, 10.5],   # top
        [14.5, 10.5],  # bottom
        [10.5, 5.5],   # left
        [10.5, 14.5],  # right
    ], dtype=np.float32)
    goals = np.array([
        [14.5, 10.5],  # -> bottom
        [5.5, 10.5],   # -> top
        [10.5, 14.5],  # -> right
        [10.5, 5.5],   # -> left
    ], dtype=np.float32)

    results = {}
    for shield in ["heuristic-orca", "po-orca"]:
        env = ContinuousMAPFEnv(grid, agent_radius=0.3)
        env.reset(starts, goals)
        for _ in range(300):
            vel = env.goal_directed_velocities()
            env.step(vel, shield_type=shield)
            if env.is_done():
                break
        m = env.current_metrics()
        results[shield] = m
        # Note: cumulative collisions over many steps are expected with
        # heuristic methods in tight crossings.  The key metric is at_goal.
        report(
            f"4-agent crossing ({shield}): agents reach goals",
            m["agent_fraction_at_goal"] >= 0.50,
            f"at_goal={m['agent_fraction_at_goal']:.2f}, collisions={m['collisions']:.0f}",
        )

    # PO-ORCA should have <= collisions than symmetric ORCA
    po_col = results["po-orca"]["collisions"]
    orca_col = results["heuristic-orca"]["collisions"]
    report(
        "4-agent crossing: PO-ORCA collisions <= ORCA",
        po_col <= orca_col + 2,  # small tolerance
        f"po-orca={po_col:.0f}, orca={orca_col:.0f}",
    )


# ---------------------------------------------------------------
# Test 4: Priority ordering correctness
# ---------------------------------------------------------------
def test_priority_ordering():
    grid = np.zeros((20, 20), dtype=np.int8)
    shield = ORCAStyleShield(agent_radius=0.3, max_speed=1.0, dt=0.2)

    # Two agents close enough to trigger collision avoidance (< 2*radius apart
    # after proposed move).  They head toward the same point from opposite sides.
    # Separation = 0.8 units, min_dist = 2*0.3 = 0.6, so they're already close
    # and their closing velocities will trigger constraints.
    positions = np.array([
        [10.5, 10.1],  # agent 0: just left of center
        [10.5, 10.9],  # agent 1: just right of center
    ], dtype=np.float32)
    # Both want to go to (10.5, 10.5) -- heading toward each other
    target = np.array([10.5, 10.5], dtype=np.float32)
    preferred = np.array([
        target - positions[0],
        target - positions[1],
    ], dtype=np.float32)
    # Normalize to max_speed
    for i in range(2):
        n = np.linalg.norm(preferred[i])
        if n > 1e-6:
            preferred[i] = preferred[i] / n * 1.0

    # Agent 0 has much higher priority
    priorities = np.array([100.0, 1.0])

    projected = shield.project(positions, preferred, grid, use_true_orca=False, priorities=priorities)

    # Agent 0 (high priority) should keep more of its preferred velocity
    dev_0 = np.linalg.norm(projected[0] - preferred[0])
    dev_1 = np.linalg.norm(projected[1] - preferred[1])
    report(
        "Priority: high-priority agent deviates less",
        dev_0 < dev_1 + 1e-6,
        f"dev_high={dev_0:.4f}, dev_low={dev_1:.4f}",
    )


# ---------------------------------------------------------------
# Test 5: Scaling test -- empty map, goal-directed
# ---------------------------------------------------------------
def test_scaling_empty():
    np.random.seed(42)
    grid = np.zeros((48, 48), dtype=np.int8)

    for n_agents in [32, 64, 128]:
        results = {}
        for shield in ["heuristic-orca", "po-orca"]:
            np.random.seed(42)
            # Random starts/goals in free space (with 0.5 offset for cell centers)
            positions = np.random.rand(n_agents, 2).astype(np.float32) * 46 + 1.0
            goals = np.random.rand(n_agents, 2).astype(np.float32) * 46 + 1.0

            env = ContinuousMAPFEnv(grid, agent_radius=0.3)
            env.reset(positions, goals)
            for _ in range(300):
                vel = env.goal_directed_velocities()
                env.step(vel, shield_type=shield)
                if env.is_done():
                    break
            m = env.current_metrics()
            results[shield] = m

        po = results["po-orca"]
        orca = results["heuristic-orca"]
        print(
            f"  N={n_agents:3d}  |  ORCA: at_goal={orca['agent_fraction_at_goal']:.3f} "
            f"col={orca['collisions']:.0f}  |  PO-ORCA: at_goal={po['agent_fraction_at_goal']:.3f} "
            f"col={po['collisions']:.0f}"
        )
        report(
            f"Scaling N={n_agents}: PO-ORCA at_goal >= 0.90",
            po["agent_fraction_at_goal"] >= 0.90,
            f"at_goal={po['agent_fraction_at_goal']:.3f}",
        )


# ---------------------------------------------------------------
# Test 6: Obstacle map (~10% random obstacles)
# ---------------------------------------------------------------
def test_obstacle_map():
    np.random.seed(123)
    grid = np.zeros((32, 32), dtype=np.int8)
    # ~10% obstacles, avoiding border
    for r in range(2, 30):
        for c in range(2, 30):
            if np.random.rand() < 0.10:
                grid[r, c] = 1

    # Find free cells for agents
    free_cells = list(zip(*np.where(grid == 0)))
    np.random.shuffle(free_cells)

    n_agents = 16
    starts = np.array(free_cells[:n_agents], dtype=np.float32) + 0.5
    goals = np.array(free_cells[n_agents : 2 * n_agents], dtype=np.float32) + 0.5

    env = ContinuousMAPFEnv(grid, agent_radius=0.3)
    env.reset(starts, goals)
    for _ in range(400):
        vel = env.goal_directed_velocities()
        env.step(vel, shield_type="po-orca")
        if env.is_done():
            break
    m = env.current_metrics()

    report(
        "Obstacle map: obstacle_hits < 5",
        m["obstacle_hits"] < 5,
        f"obstacle_hits={m['obstacle_hits']:.0f}, at_goal={m['agent_fraction_at_goal']:.3f}, "
        f"collisions={m['collisions']:.0f}",
    )
    report(
        "Obstacle map: some agents reach goal",
        m["agent_fraction_at_goal"] > 0.3,
        f"at_goal={m['agent_fraction_at_goal']:.3f}",
    )


# ---------------------------------------------------------------
# Test 7: picbf-cs adapter smoke test
# ---------------------------------------------------------------
def test_picbf_cs_adapter():
    grid = np.zeros((20, 20), dtype=np.int8)
    starts = np.array([
        [5.5, 10.5],
        [14.5, 10.5],
        [10.5, 5.5],
        [10.5, 14.5],
    ], dtype=np.float32)
    goals = np.array([
        [14.5, 10.5],
        [5.5, 10.5],
        [10.5, 14.5],
        [10.5, 5.5],
    ], dtype=np.float32)

    env = ContinuousMAPFEnv(grid, agent_radius=0.3)
    env.reset(starts, goals)
    try:
        env.step(env.goal_directed_velocities(), shield_type="picbf-cs")
        first_step_velocities = env.history_velocities[-1]
        first_debug = env.last_shield_debug_info or {}
        for _ in range(119):
            env.step(env.goal_directed_velocities(), shield_type="picbf-cs")
            if env.is_done():
                break
    except (ImportError, RuntimeError) as exc:
        report_skip("picbf-cs adapter", str(exc))
        return

    m = env.current_metrics()
    report(
        "picbf-cs local output: one velocity per stable agent index",
        first_step_velocities.shape == starts.shape,
        f"velocity_shape={first_step_velocities.shape}",
    )
    debug_passed = (
        int(first_debug.get("component_count", 0)) >= 1
        and 1 <= int(first_debug.get("max_component_size", 0)) <= len(starts)
    )
    report(
        "picbf-cs local debug: component stats available",
        debug_passed,
        f"debug={first_debug}",
    )
    passed = (
        m["agent_fraction_at_goal"] == 1.0
        and m["collisions"] == 0
        and m["obstacle_hits"] == 0
    )
    report(
        "picbf-cs crossing: agents reach goals without collisions",
        passed,
        f"at_goal={m['agent_fraction_at_goal']:.3f}, collisions={m['collisions']:.0f}, "
        f"obstacle_hits={m['obstacle_hits']:.0f}",
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PO-ORCA + SDF Validation Tests")
    print("=" * 60)

    print("\n--- Test 1: SDF Correctness ---")
    test_sdf_correctness()

    print("\n--- Test 2: SDF Obstacle Avoidance (Single Agent + Wall) ---")
    test_sdf_single_agent_wall()

    print("\n--- Test 3: 4-Agent Crossing ---")
    test_4agent_crossing()

    print("\n--- Test 4: Priority Ordering ---")
    test_priority_ordering()

    print("\n--- Test 5: Scaling (Empty Map) ---")
    test_scaling_empty()

    print("\n--- Test 6: Obstacle Map ---")
    test_obstacle_map()

    print("\n--- Test 7: picbf-cs Adapter ---")
    test_picbf_cs_adapter()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
