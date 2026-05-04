#!/usr/bin/env python3
"""Generate interpretation-first ablation evidence artifacts."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from report_data import (
    ABLATION_OBSERVATIONS,
    FIG_DIR,
    TABLE_DIR,
    ensure_dirs,
    latex_escape,
    write_csv,
    write_tex_table,
)


def write_inference_table() -> None:
    write_csv(
        TABLE_DIR / "ablation_inference_matrix.csv",
        ABLATION_OBSERVATIONS,
        ["factor", "evidence", "target_effect", "broad_effect", "inference"],
    )
    write_tex_table(
        TABLE_DIR / "ablation_inference_matrix.tex",
        [
            [
                latex_escape(row["factor"]),
                latex_escape(row["target_effect"]),
                latex_escape(row["broad_effect"]),
                latex_escape(row["inference"]),
            ]
            for row in ABLATION_OBSERVATIONS
        ],
        ["Intervention", "Targeted topology", "Broader evaluation", "Supported inference"],
        "p{0.21\\linewidth}p{0.20\\linewidth}p{0.20\\linewidth}p{0.31\\linewidth}",
    )


def write_inference_map() -> None:
    path = FIG_DIR / "ablation_inference_map.tex"
    with path.open("w") as f:
        f.write("\\begin{tikzpicture}[\n")
        f.write("  font=\\scriptsize,\n")
        f.write("  evidence/.style={rectangle,draw=black!45,rounded corners=1.5pt,")
        f.write("fill=white,align=left,text width=3.95cm,minimum height=1.02cm,inner sep=4pt},\n")
        f.write("  finding/.style={rectangle,draw=black!55,rounded corners=1.5pt,")
        f.write("fill=oiBlue!7,align=left,text width=4.10cm,minimum height=1.05cm,inner sep=4pt},\n")
        f.write("  center/.style={rectangle,draw=black!60,rounded corners=1.5pt,")
        f.write("fill=oiVermillion!8,align=center,text width=3.25cm,minimum height=1.35cm,inner sep=5pt},\n")
        f.write("  arr/.style={-Stealth,thick,draw=black!55}\n")
        f.write("]\n")

        left_x = 0.0
        center_x = 5.1
        right_x = 10.2
        ys = [5.4, 3.75, 2.10, 0.45]
        labels = [
            (
                "Action-interface alignment",
                "0.00\\% initial diagnostic evaluation to 43.48\\% with matched flow state; "
                "62.70\\% in the larger diagnostic run.",
                "oiOrange!15",
            ),
            (
                "Discrete-primary supervision",
                "Dense-random target reached 75\\% in one setting, while broad "
                "diagnostic success stayed near 60--61\\%.",
                "oiBluishGreen!13",
            ),
            (
                "Coordination-focused data",
                "Strict flow-only control reached 70\\% on the targeted dense-random "
                "map, but broad transfer stayed near 60.5--62.4\\%.",
                "oiSkyBlue!16",
            ),
            (
                "Hybrid action policy",
                "Primary held-out evaluation: 62.49\\% overall and 58\\% dense-random, "
                "below the velocity policy's 64.59\\% and 71\\%.",
                "oiReddishPurple!13",
            ),
        ]

        for idx, (title, body, fill) in enumerate(labels):
            y = ys[idx]
            f.write(
                f"\\node[evidence,fill={fill}] (e{idx}) at ({left_x:.2f},{y:.2f}) "
                f"{{\\textbf{{{title}}}\\\\{body}}};\n"
            )

        f.write(
            f"\\node[center] (gap) at ({center_x:.2f},3.05) "
            "{Heavy-coordination gap\\\\requires wait/yield timing and bottleneck negotiation};\n"
        )
        findings = [
            (
                "f0",
                5.05,
                "Discrete action supervision appears useful on coordination-heavy maps.",
                "oiBluishGreen!10",
            ),
            (
                "f1",
                3.05,
                "The velocity-to-action bridge remains a plausible bottleneck.",
                "oiVermillion!8",
            ),
            (
                "f2",
                1.05,
                "The evidence does not prove that flow-based policies cannot learn coordination.",
                "black!4",
            ),
        ]
        for name, y, text, fill in findings:
            f.write(
                f"\\node[finding,fill={fill}] ({name}) at ({right_x:.2f},{y:.2f}) "
                f"{{{text}}};\n"
            )

        for idx in range(4):
            f.write(f"\\draw[arr] (e{idx}.east) -- (gap.west);\n")
        f.write("\\draw[arr] (gap.east) -- (f0.west);\n")
        f.write("\\draw[arr] (gap.east) -- (f1.west);\n")
        f.write("\\draw[arr,densely dashed] (gap.east) -- (f2.west);\n")

        f.write(
            "\\node[rectangle,draw=black!45,fill=black!3,rounded corners=1.5pt,"
            "align=center,text width=13.2cm,inner sep=4pt] "
            "at (5.1,-1.10) {Ablations are used as evidence about interface, "
            "objective, and data-distribution effects; none is treated as a new "
            "state-of-the-art method.};\n"
        )
        f.write("\\end{tikzpicture}\n")


def main() -> int:
    ensure_dirs()
    write_inference_table()
    write_inference_map()
    print(f"Wrote ablation inference map to {FIG_DIR / 'ablation_inference_map.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
