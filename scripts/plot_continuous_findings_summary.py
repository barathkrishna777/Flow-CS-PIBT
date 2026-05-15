#!/usr/bin/env python3
"""Generate dependency-free SVG plots for continuous-space MAPF findings."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

COLORS = {
    "orca": "#999999",
    "po_orca": "#CC79A7",
    "straight_epibt": "#0072B2",
    "flow_orca": "#E69F00",
    "flow_epibt": "#009E73",
    "grid": "#d9d9d9",
    "text": "#222222",
}


def text(x, y, value, size=13, anchor="middle", weight="400", color=None):
    color = color or COLORS["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(str(value))}</text>'
    )


def rect(x, y, w, h, fill, stroke="#222222", sw=0.8):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1, y1, x2, y2, color="#d9d9d9", sw=1.0):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw}"/>'
    )


def bar_panel(x0, y0, w, h, labels, values, colors, y_max, title, y_label):
    out = []
    out.append(text(x0 + w / 2, y0 - 18, title, 15, weight="700"))
    out.append(text(x0 - 46, y0 + h / 2, y_label, 12, anchor="middle"))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + h - frac * h
        out.append(line(x0, y, x0 + w, y))
        out.append(text(x0 - 8, y + 4, f"{int(frac * y_max)}", 10, anchor="end", color="#555555"))
    out.append(line(x0, y0, x0, y0 + h, "#333333", 1.1))
    out.append(line(x0, y0 + h, x0 + w, y0 + h, "#333333", 1.1))

    n = len(values)
    gap = 10
    bw = (w - gap * (n + 1)) / n
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        bh = max(0, val / y_max * h)
        bx = x0 + gap + i * (bw + gap)
        by = y0 + h - bh
        out.append(rect(bx, by, bw, bh, color))
        shown = "~0" if val == 0 else (f"{val:.1f}" if y_max <= 100 else f"{int(val)}")
        out.append(text(bx + bw / 2, max(y0 + 12, by - 5), shown, 10))
        for row, part in enumerate(label.split("\n")):
            out.append(text(bx + bw / 2, y0 + h + 18 + 13 * row, part, 10))
    return out


def primary_decomposition():
    labels = ["ORCA", "PO-ORCA", "Straight\n+CV-PIBT", "Flow\n+ORCA", "Flow\n+CV-PIBT"]
    at_goal = [16.8, 14.9, 77.4, 48.3, 87.2]
    collisions = [4169, 2626, 1307, 0, 653]
    colors = [
        COLORS["orca"],
        COLORS["po_orca"],
        COLORS["straight_epibt"],
        COLORS["flow_orca"],
        COLORS["flow_epibt"],
    ]

    width, height = 950, 360
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 28, "Primary dense random setting, N=100, 512 steps", 17, weight="700"),
    ]
    parts += bar_panel(72, 72, 365, 210, labels, at_goal, colors, 100, "Agent completion", "Percent")
    parts += bar_panel(548, 72, 365, 210, labels, collisions, colors, 4500, "Empirical collisions", "Count")
    parts.append(text(255, 340, "Shielding: ORCA 16.8% -> Straight+CV-PIBT 77.4%", 12, color="#444444"))
    parts.append(text(735, 340, "Learning: Straight+CV-PIBT 77.4% -> Flow+CV-PIBT 87.2%", 12, color="#444444"))
    parts.append("</svg>")
    return "\n".join(parts)


def generalization():
    maps = ["Random-64\nnear dist.", "Room\nnear dist.", "Warehouse\nOOD"]
    series = [
        ("ORCA", [4.8, 1.2, 13.2], COLORS["orca"]),
        ("Straight + CV-PIBT", [50.5, 11.4, 41.1], COLORS["straight_epibt"]),
        ("Flow + CV-PIBT", [61.5, 26.5, 25.4], COLORS["flow_epibt"]),
    ]
    width, height = 820, 360
    x0, y0, w, h = 82, 70, 650, 210
    y_max = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 28, "Generalization of shielding and learned guidance, N=50", 17, weight="700"),
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + h - frac * h
        parts.append(line(x0, y, x0 + w, y))
        parts.append(text(x0 - 8, y + 4, f"{int(frac * y_max)}", 10, anchor="end", color="#555555"))
    parts.append(line(x0, y0, x0, y0 + h, "#333333", 1.1))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, "#333333", 1.1))
    parts.append(text(24, y0 + h / 2, "Agents at goal (%)", 12))

    group_w = w / len(maps)
    bar_w = 48
    for gi, map_label in enumerate(maps):
        cx = x0 + group_w * gi + group_w / 2
        for si, (_, vals, color) in enumerate(series):
            bx = cx + (si - 1) * (bar_w + 5) - bar_w / 2
            val = vals[gi]
            bh = val / y_max * h
            by = y0 + h - bh
            parts.append(rect(bx, by, bar_w, bh, color))
            parts.append(text(bx + bar_w / 2, by - 5, f"{val:.1f}", 10))
        for row, part in enumerate(map_label.split("\n")):
            parts.append(text(cx, y0 + h + 18 + 13 * row, part, 10))

    lx = 140
    for i, (label, _, color) in enumerate(series):
        x = lx + i * 190
        parts.append(rect(x, 46, 14, 14, color))
        parts.append(text(x + 22, 58, label, 11, anchor="start"))
    parts.append(text(width / 2, 340, "Flow guidance helps on near-distribution maps; warehouse exposes OOD learned-policy gap.", 12, color="#444444"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "continuous_primary_decomposition.svg": primary_decomposition(),
        "continuous_generalization_completion.svg": generalization(),
    }
    for name, svg in files.items():
        (OUT / name).write_text(svg, encoding="utf-8")
    print(f"Wrote {len(files)} SVG figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
