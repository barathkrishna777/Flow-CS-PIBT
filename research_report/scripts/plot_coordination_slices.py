#!/usr/bin/env python3
"""Generate a compact topology-gap figure for coordination-heavy benchmark gaps."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from report_data import FIG_DIR, TABLE_DIR, ensure_dirs, get_summary, latex_escape, write_csv


PRIMARY_MAP_REGIMES = [
    {
        "map": "Paris_1_256",
        "label": "Paris",
        "topology": "Open/city",
        "color": "oiBlue",
    },
    {
        "map": "den520d",
        "label": "den520d",
        "topology": "Open/game",
        "color": "oiBlue",
    },
    {
        "map": "empty-48-48",
        "label": "empty",
        "topology": "Open",
        "color": "oiBlue",
    },
    {
        "map": "random-64-64-10",
        "label": "random-64",
        "topology": "Moderate random",
        "color": "oiBluishGreen",
    },
    {
        "map": "den312d",
        "label": "den312d",
        "topology": "Cluttered game",
        "color": "oiBluishGreen",
    },
    {
        "map": "random-32-32-10",
        "label": "dense random",
        "topology": "Dense random",
        "color": "oiVermillion",
    },
    {
        "map": "maze-128-128-2",
        "label": "maze bottleneck",
        "topology": "Bottleneck",
        "color": "oiVermillion",
    },
    {
        "map": "warehouse-10-20-10-2-1",
        "label": "warehouse",
        "topology": "Floor case",
        "color": "black!55",
    },
]


def build_rows() -> list[dict[str, object]]:
    flow = get_summary("rishi8_flow")
    ssil = get_summary("rishi8_ssil")
    rows: list[dict[str, object]] = []
    for item in PRIMARY_MAP_REGIMES:
        map_name = item["map"]
        flow_success = float(flow["per_map"][map_name])
        ssil_success = float(ssil["per_map"][map_name])
        rows.append(
            {
                **item,
                "flow_success": flow_success,
                "ssil_success": ssil_success,
                "ssil_minus_flow": ssil_success - flow_success,
            }
        )
    return rows


def write_topology_table(rows: list[dict[str, object]]) -> None:
    write_csv(
        TABLE_DIR / "topology_coordination_regime.csv",
        rows,
        ["map", "label", "topology", "flow_success", "ssil_success", "ssil_minus_flow"],
    )


def write_topology_figure(rows: list[dict[str, object]]) -> None:
    path = FIG_DIR / "topology_coordination_regime.tex"
    x_min = -5.0
    x_max = 12.0
    shift = -x_min
    n = len(rows)
    with path.open("w") as f:
        f.write("\\begin{tikzpicture}[x=0.42cm,y=0.48cm]\n")
        f.write("\\scriptsize\n")
        f.write("\\node[anchor=west,font=\\bfseries] at (-9.1,9.1) {Topology};\n")
        f.write("\\node[anchor=west,font=\\bfseries] at (-4.7,9.1) {Map};\n")
        f.write("\\node[anchor=east,font=\\bfseries] at (25.2,9.1) {FLOMAP / SSIL};\n")
        f.write("\\draw[black!35] (-9.1,8.72) -- (25.6,8.72);\n")
        zero = shift
        f.write(f"\\draw[black!45] ({zero:.2f},0.45) -- ({zero:.2f},{n + 0.25:.2f});\n")
        for tick in [-5, 0, 5, 10]:
            x = tick + shift
            f.write(
                f"\\draw[black!20] ({x:.2f},0.38) -- ({x:.2f},{n + 0.25:.2f});"
                f"\\node[below,black!70] at ({x:.2f},0.25) {{{tick}}};\n"
            )
        f.write(
            f"\\node[below,black!80] at ({(x_max - x_min) / 2:.2f},-0.38) "
            "{Learned-policy success-rate difference (pp)};\n"
        )
        f.write("\\node[anchor=east,black!60] at (4.65,0.95) {Higher FLOMAP success};\n")
        f.write("\\node[anchor=west,black!60] at (5.35,0.95) {Higher SSIL success};\n")

        for idx, row in enumerate(rows):
            y = n - idx
            gap = float(row["ssil_minus_flow"])
            x = gap + shift
            color = str(row["color"])
            label = latex_escape(str(row["label"]))
            topo = latex_escape(str(row["topology"]))
            flow = float(row["flow_success"])
            ssil = float(row["ssil_success"])
            f.write(f"\\draw[black!10] (-9.1,{y - 0.42:.2f}) -- (25.6,{y - 0.42:.2f});\n")
            f.write(f"\\node[anchor=west,black!78] at (-9.1,{y:.2f}) {{{topo}}};\n")
            f.write(f"\\node[anchor=west,black!88] at (-4.7,{y:.2f}) {{{label}}};\n")
            f.write(f"\\draw[black!35,line width=0.45pt] ({zero:.2f},{y:.2f}) -- ({x:.2f},{y:.2f});\n")
            f.write(f"\\filldraw[fill={color},draw=black!45] ({x:.2f},{y:.2f}) circle[radius=0.10];\n")
            f.write(f"\\node[anchor=east,black!78] at (25.2,{y:.2f}) {{{flow:.1f} / {ssil:.1f}}};\n")

        f.write("\\end{tikzpicture}\n")


def main() -> int:
    ensure_dirs()
    rows = build_rows()
    write_topology_table(rows)
    write_topology_figure(rows)
    print(f"Wrote topology-regime figure to {FIG_DIR / 'topology_coordination_regime.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
