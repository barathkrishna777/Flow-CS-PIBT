import os
import re
from typing import Iterable, List, Optional, Sequence


_SCENARIO_RE = re.compile(r"-random-(\d+)\.scen$")


def scenario_id_from_path(path: str) -> int:
    name = os.path.basename(path)
    match = _SCENARIO_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse scenario id from path: {path}")
    return int(match.group(1))


def sort_scenarios_numerically(paths: Iterable[str]) -> List[str]:
    return sorted(paths, key=lambda path: (scenario_id_from_path(path), os.path.basename(path)))


def select_scenarios(
    paths: Sequence[str],
    max_scenarios: Optional[int] = None,
    scenario_ids: Optional[Sequence[int]] = None,
    scenario_start: Optional[int] = None,
    scenario_end: Optional[int] = None,
) -> List[str]:
    selected = sort_scenarios_numerically(paths)
    allowed_ids = set(int(v) for v in scenario_ids) if scenario_ids else None
    filtered: List[str] = []
    for path in selected:
        sid = scenario_id_from_path(path)
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        if scenario_start is not None and sid < scenario_start:
            continue
        if scenario_end is not None and sid > scenario_end:
            continue
        filtered.append(path)
    if max_scenarios is not None and max_scenarios > 0:
        filtered = filtered[:max_scenarios]
    return filtered
