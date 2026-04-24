import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../../..");
const TEMPLATE_PATH = path.join(ROOT, "powerpoint-template.potx");
const OUT_DIR = path.join(ROOT, "presentation", "advisor_update", "outputs");
const MEDIA_DIR = path.join(ROOT, "presentation", "advisor_update", "media");
const DATA_PATH = path.join(ROOT, "presentation", "advisor_update", "generated_plots", "grid_summary.json");
const PREVIEW_DIR = path.join(ROOT, "presentation", "advisor_update", "tmp", "previews");
const SPEAKER_NOTES_PATH = path.join(ROOT, "presentation", "advisor_update", "speaker_notes.md");
const OUTPUT_PPTX = path.join(OUT_DIR, "flow_cs_pibt_advisor_update_2026-04-24.pptx");

const W = 960;
const H = 540;

const CMU_RED = "#BB0000";
const CMU_NAVY = "#002C71";
const CMU_GREEN = "#00833C";
const CMU_GOLD = "#F2A900";
const CMU_GRAY = "#75787B";
const LIGHT_GRAY = "#E8E7E3";
const SOFT_RED = "#F9ECEC";
const SOFT_NAVY = "#EEF2F8";
const SOFT_GREEN = "#EDF6F0";
const SOFT_GOLD = "#FFF6E2";
const TEXT = "#24303A";
const MUTED = "#5D6770";
const WHITE = "#FFFFFF";
const CLEAR = "#00000000";
const MIN_TEXT = 22;

const TITLE_FACE = "Arial";
const BODY_FACE = "Arial";
const MONO_FACE = "Arial";

const notes = [];

function prettyLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, geometry, left, top, width, height, fill = CLEAR, line = CLEAR, lineWidth = 0) {
  return slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: lineWidth },
  });
}

function addText(
  slide,
  text,
  left,
  top,
  width,
  height,
  {
    size = MIN_TEXT,
    color = TEXT,
    bold = false,
    face = BODY_FACE,
    align = "left",
    valign = "top",
    fill = CLEAR,
    line = CLEAR,
    lineWidth = 0,
    autoFit = "shrinkText",
  } = {},
) {
  const box = addShape(slide, "rect", left, top, width, height, fill, line, lineWidth);
  box.text = text;
  box.text.fontSize = size;
  box.text.color = color;
  box.text.bold = bold;
  box.text.typeface = face;
  box.text.alignment = align;
  box.text.verticalAlignment = valign;
  box.text.insets = { left: 6, right: 6, top: 4, bottom: 4 };
  if (autoFit) {
    box.text.autoFit = autoFit;
  }
  return box;
}

async function addImage(slide, imagePath, left, top, width, height, fit = "contain", geometry = null) {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt: path.basename(imagePath),
  });
  image.position = { left, top, width, height };
  if (geometry) {
    image.geometry = geometry;
  }
  return image;
}

function addSectionHeader(slide, section, title, subtitle, slideNo) {
  addText(slide, title, 40, 32, 780, 44, {
    size: 34,
    color: TEXT,
    bold: true,
    face: TITLE_FACE,
  });
  if (subtitle) {
    addText(slide, subtitle, 42, 78, 840, 28, {
      size: MIN_TEXT,
      color: MUTED,
      face: BODY_FACE,
      autoFit: null,
    });
  }
  addShape(slide, "rect", 40, 118, 878, 2, CMU_RED, CLEAR, 0);
}

function addBulletBlock(slide, lines, left, top, width, height, color = TEXT, size = 18) {
  const text = lines.map((line) => `- ${line}`).join("\n\n");
  addText(slide, text, left, top, width, height, {
    size,
    color,
    face: BODY_FACE,
    autoFit: "shrinkText",
  });
}

function addCard(slide, left, top, width, height, title, body, accent = CMU_RED, fill = WHITE, options = {}) {
  const {
    compact = false,
    titleSize = 22,
    bodySize = 22,
    titleHeight = compact ? 26 : 28,
    bodyTop = compact ? 42 : 46,
    padX = compact ? 12 : 14,
    padTop = compact ? 12 : 14,
  } = options;
  addShape(slide, "roundRect", left, top, width, height, fill, accent, 1.2);
  addShape(slide, "rect", left, top, width, 6, accent, CLEAR, 0);
  addText(slide, title, left + padX, top + padTop, width - padX * 2, titleHeight, {
    size: titleSize,
    color: accent,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  addText(slide, body, left + padX, top + bodyTop, width - padX * 2, Math.max(12, height - bodyTop - 10), {
    size: bodySize,
    color: TEXT,
    face: BODY_FACE,
    autoFit: null,
  });
}

function addBanner(slide, left, top, width, height, text, accent = CMU_RED, fill = WHITE) {
  addShape(slide, "roundRect", left, top, width, height, fill, accent, 1.2);
  addShape(slide, "rect", left, top, width, 6, accent, CLEAR, 0);
  addText(slide, text, left + 16, top + 12, width - 32, height - 18, {
    size: MIN_TEXT,
    color: accent,
    bold: true,
    face: BODY_FACE,
    valign: "middle",
    autoFit: null,
  });
}

function addMetricCard(slide, left, top, width, height, value, label, note = null, accent = CMU_RED) {
  addShape(slide, "roundRect", left, top, width, height, WHITE, accent, 1.2);
  addShape(slide, "rect", left, top, width, 7, accent, CLEAR, 0);
  addText(slide, value, left + 14, top + 16, width - 28, 28, {
    size: 30,
    color: TEXT,
    bold: true,
    face: TITLE_FACE,
  });
  addText(slide, label, left + 14, top + 52, width - 28, 28, {
    size: MIN_TEXT,
    color: MUTED,
    face: BODY_FACE,
    autoFit: null,
  });
  if (note) {
    addText(slide, note, left + 14, top + 80, width - 28, height - 90, {
      size: MIN_TEXT,
      color: MUTED,
      face: BODY_FACE,
    });
  }
}

function addDividerArrow(slide, left, top, width, height, fill = CMU_RED) {
  addShape(slide, "rightArrow", left, top, width, height, fill, fill, 0.5);
}

function addTable(slide, left, top, widths, rows, headerFill = CMU_NAVY) {
  const rowHeight = 28;
  const totalWidth = widths.reduce((sum, value) => sum + value, 0);
  addShape(slide, "roundRect", left, top, totalWidth, rowHeight * rows.length, WHITE, LIGHT_GRAY, 1);
  let x = left;
  for (let i = 0; i < widths.length; i += 1) {
    addShape(slide, "rect", x, top, widths[i], rowHeight, headerFill, CLEAR, 0);
    x += widths[i];
  }
  rows.forEach((row, rowIndex) => {
    const y = top + rowHeight * rowIndex;
    if (rowIndex > 0) {
      addShape(slide, "rect", left, y, totalWidth, 1, LIGHT_GRAY, CLEAR, 0);
    }
    let colX = left;
    row.forEach((cell, colIndex) => {
      if (colIndex > 0) {
        addShape(slide, "rect", colX, y, 1, rowHeight, LIGHT_GRAY, CLEAR, 0);
      }
      addText(slide, cell, colX + 6, y + 4, widths[colIndex] - 12, rowHeight - 8, {
        size: rowIndex === 0 ? 11 : 10,
        color: rowIndex === 0 ? WHITE : TEXT,
        bold: rowIndex === 0,
        face: rowIndex === 0 ? MONO_FACE : BODY_FACE,
        valign: "middle",
        autoFit: "shrinkText",
      });
      colX += widths[colIndex];
    });
  });
}

function styleChart(chart, title, legend = true) {
  chart.title = title;
  chart.hasLegend = legend;
  if (legend) {
    chart.legend.position = "bottom";
  }
  if (chart.titleTextStyle) {
    chart.titleTextStyle.fontSize = MIN_TEXT;
    chart.titleTextStyle.fill = TEXT;
    chart.titleTextStyle.typeface = TITLE_FACE;
  }
  if (chart.legend?.textStyle) {
    chart.legend.textStyle.fontSize = MIN_TEXT;
    chart.legend.textStyle.typeface = BODY_FACE;
  }
  if (chart.xAxis?.textStyle) {
    chart.xAxis.textStyle.fontSize = MIN_TEXT;
    chart.xAxis.textStyle.typeface = BODY_FACE;
  }
  if (chart.yAxis?.textStyle) {
    chart.yAxis.textStyle.fontSize = MIN_TEXT;
    chart.yAxis.textStyle.typeface = BODY_FACE;
  }
  if (chart.dataLabels?.textStyle) {
    chart.dataLabels.textStyle.fontSize = MIN_TEXT;
    chart.dataLabels.textStyle.typeface = BODY_FACE;
  }
}

function addLineChart(slide, left, top, width, height, title, categories, seriesDefs) {
  addShape(slide, "roundRect", left, top, width, height, WHITE, LIGHT_GRAY, 1);
  const chart = slide.charts.add("line");
  chart.position = { left: left + 10, top: top + 8, width: width - 20, height: height - 16 };
  chart.categories = categories;
  chart.lineOptions.grouping = "standard";
  chart.lineOptions.smooth = false;
  styleChart(chart, title, true);
  seriesDefs.forEach((def) => {
    const series = chart.series.add(def.name);
    series.values = def.values;
    series.categories = categories;
    series.stroke = { width: 2.5, style: "solid", fill: def.color };
    series.fill = def.color;
  });
  return chart;
}

function addBarChart(slide, left, top, width, height, title, categories, seriesDefs) {
  addShape(slide, "roundRect", left, top, width, height, WHITE, LIGHT_GRAY, 1);
  const chart = slide.charts.add("bar");
  chart.position = { left: left + 10, top: top + 8, width: width - 20, height: height - 16 };
  chart.categories = categories;
  chart.barOptions.direction = "column";
  chart.barOptions.grouping = seriesDefs.length > 1 ? "clustered" : "standard";
  chart.dataLabels.showValue = true;
  chart.dataLabels.position = "outEnd";
  styleChart(chart, title, seriesDefs.length > 1);
  seriesDefs.forEach((def) => {
    const series = chart.series.add(def.name);
    series.values = def.values;
    series.categories = categories;
    series.fill = def.color;
    series.stroke = { width: 1.2, style: "solid", fill: def.color };
  });
  return chart;
}

function addMiniGridMatrix(slide, left, top, cols, rows, cellW, cellH) {
  addShape(slide, "roundRect", left - 12, top - 16, cols * cellW + 68, rows * cellH + 52, SOFT_NAVY, LIGHT_GRAY, 1);
  addText(slide, "Eval matrix", left, top - 6, 120, 16, {
    size: 11,
    color: CMU_NAVY,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  for (let c = 0; c < cols; c += 1) {
    addText(slide, `${(c + 1) * 100}`, left + 52 + c * cellW, top + rows * cellH + 6, cellW - 4, 12, {
      size: 8,
      color: MUTED,
      face: MONO_FACE,
      align: "center",
      autoFit: null,
    });
  }
  for (let r = 0; r < rows; r += 1) {
    addText(slide, `${r + 1}`, left, top + r * cellH + 2, 24, cellH - 2, {
      size: 8,
      color: MUTED,
      face: MONO_FACE,
      align: "right",
      autoFit: null,
    });
    for (let c = 0; c < cols; c += 1) {
      const fill = (r + c) % 2 === 0 ? "#D7E2F3" : "#F7D9D9";
      addShape(slide, "roundRect", left + 34 + c * cellW, top + r * cellH, cellW - 5, cellH - 4, fill, CLEAR, 0);
    }
  }
  addText(slide, "rows: maps x scenarios", left, top + rows * cellH + 20, 120, 12, {
    size: 8,
    color: MUTED,
    face: BODY_FACE,
    autoFit: null,
  });
}

function setNotes(slide, slideNo, title, bullets, sources = []) {
  notes.push({ slideNo, title, bullets, sources });
  const body = bullets.map((line) => `- ${line}`).join("\n");
  const sourceText = sources.length ? `\n\nSources:\n${sources.map((src) => `- ${src}`).join("\n")}` : "";
  slide.speakerNotes.setText(`${title}\n\n${body}${sourceText}`);
}

function findShapeWithText(slide, fragment) {
  return slide.shapes.items.find((shape) => {
    const text = shape.text?.toString?.() || "";
    return text.includes(fragment);
  });
}

function buildTitleSlide(slide) {
  const titleShape = findShapeWithText(slide, "Presentation Title");
  const presenterShape = findShapeWithText(slide, "Presenter Name");
  if (!titleShape || !presenterShape) {
    throw new Error("Could not find title placeholders in template slide.");
  }

  titleShape.text = "Flow-CS-PIBT Research Update";
  titleShape.text.fontSize = 36;
  titleShape.text.bold = true;
  titleShape.text.color = WHITE;
  titleShape.text.typeface = TITLE_FACE;
  titleShape.position = { left: 186, top: 168, width: 610, height: 92 };

  presenterShape.text = "Grid-world MAPF results and next steps\nBarath Krishna | April 24, 2026";
  presenterShape.text.fontSize = MIN_TEXT;
  presenterShape.text.color = WHITE;
  presenterShape.text.typeface = BODY_FACE;
  presenterShape.position = { left: 186, top: 278, width: 610, height: 78 };

  addShape(slide, "roundRect", 186, 378, 304, 54, "#A30000CC", WHITE, 0.8);
  addText(slide, "Main focus: barath/flow-gnn", 202, 392, 272, 24, {
    size: MIN_TEXT,
    color: WHITE,
    bold: true,
    face: MONO_FACE,
    align: "center",
    valign: "middle",
    autoFit: null,
  });

  setNotes(
    slide,
    1,
    "Flow-CS-PIBT Research Update",
    [
      "Frame this as a research update, not a final paper pitch.",
      "The talk centers on the grid-world branch results, then uses those results to motivate the continuous-space and transformer directions.",
      "The headline tension is simple: flow is promising, but the discrete planner interface is still the bottleneck.",
    ],
    ["powerpoint-template.potx", "README.md", "PAPER_PLAN.md"],
  );
}

async function buildMotivation(slide, slideNo) {
  addSectionHeader(slide, "Motivation", "Why local MAPF still matters", "Good local motion is not enough; the planner interface matters too.", slideNo);
  addCard(slide, 46, 148, 350, 82, "Why MAPF", "Search cost rises with density.", CMU_RED, SOFT_RED);
  addCard(slide, 46, 246, 350, 82, "Why grid-world", "It isolates shielding cleanly.", CMU_NAVY, SOFT_NAVY);
  addCard(slide, 46, 344, 350, 82, "This repo's question", "Can policy help the shield?", CMU_GREEN, SOFT_GREEN);
  await addImage(slide, path.join(MEDIA_DIR, "motivation_montage.png"), 430, 146, 478, 282, "contain");
  addBanner(slide, 46, 438, 862, 48, "Grid-world testbed: same maps, same scenarios, same CS-PIBT shield.", CMU_NAVY, SOFT_NAVY);
  setNotes(
    slide,
    slideNo,
    "Why learned local MAPF still matters",
    [
      "Open by separating two failure sources: finding a locally sensible direction versus resolving conflicts under a decentralized shield.",
      "Grid-world is still the right place to debug that interface because the state, action set, and benchmark protocol are controlled.",
      "That is why the rest of the deck uses the held-out Rishi benchmark as the main reference point.",
    ],
    ["README.md", "PROJECT_CONTEXT_HANDOFF.md", "logs/slide_gifs/*"],
  );
}

function buildRelated(slide, slideNo) {
  addSectionHeader(slide, "Related Work", "Closest prior work: SSIL + CS-PIBT", "Same benchmark, different policy output.", slideNo);
  addCard(slide, 50, 150, 260, 118, "SSIL", "Predict the next 5-way move.", CMU_NAVY, SOFT_NAVY);
  addCard(slide, 350, 150, 260, 118, "Flow-CS", "Predict a 2D direction field.", CMU_RED, SOFT_RED);
  addCard(slide, 650, 150, 260, 118, "Shared setup", "Same observations, same shield.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 92, 320, 776, 106, "Why Rishi matters here", "It sets the right question: can better geometry help while keeping the same planner?", CMU_GOLD, SOFT_GOLD);
  setNotes(
    slide,
    slideNo,
    "Closest prior work: SSIL + CS-PIBT",
    [
      "Emphasize that we are not discarding the successful CS-PIBT line; we are changing the policy object that feeds it.",
      "That makes the comparison fair: same shield, same maps, same scenario ladder, same success criterion.",
      "The key hypothesis is that flow should be a better inductive bias for geometry and later continuous-space transfer, even if the current grid interface is imperfect.",
    ],
    ["README.md", "PAPER_PLAN.md", "FLOW_VS_SSIL_ABLATION_PLAN.md"],
  );
}

function buildProblem(slide, slideNo) {
  addSectionHeader(slide, "Formulation", "Problem setup and target interface", "Each agent sees a local neighborhood; the shield still needs a ranking.", slideNo);
  addShape(slide, "roundRect", 46, 150, 320, 250, WHITE, CMU_NAVY, 1.2);
  addText(slide, "Local state", 64, 166, 160, 26, {
    size: MIN_TEXT,
    color: CMU_NAVY,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  const gridLeft = 74;
  const gridTop = 214;
  const cell = 30;
  for (let r = 0; r < 7; r += 1) {
    for (let c = 0; c < 7; c += 1) {
      let fill = "#FAF9F5";
      if ((r === 1 && c === 3) || (r === 2 && c === 3) || (r === 4 && c === 1) || (r === 4 && c === 2) || (r === 4 && c === 3)) {
        fill = CMU_GRAY;
      }
      addShape(slide, "rect", gridLeft + c * cell, gridTop + r * cell, cell - 2, cell - 2, fill, LIGHT_GRAY, 0.5);
    }
  }
  addShape(slide, "ellipse", gridLeft + 1 * cell + 3, gridTop + 1 * cell + 3, 22, 22, CMU_RED, WHITE, 1.2);
  addShape(slide, "ellipse", gridLeft + 5 * cell + 3, gridTop + 5 * cell + 3, 22, 22, CMU_GREEN, WHITE, 1.2);
  addShape(slide, "ellipse", gridLeft + 5 * cell + 6, gridTop + 1 * cell + 6, 14, 14, CMU_GOLD, WHITE, 0.8);
  addShape(slide, "ellipse", gridLeft + 1 * cell + 6, gridTop + 5 * cell + 6, 14, 14, CMU_GOLD, WHITE, 0.8);
  addBanner(slide, 46, 418, 320, 68, "Input: patch, occupancy, BD, nearby agents.", CMU_NAVY, SOFT_NAVY);

  addCard(slide, 404, 154, 504, 84, "Objective", "Maximize all-agents success.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 404, 254, 504, 84, "Policy output", "Velocity vector or 5-way logits.", CMU_GOLD, SOFT_GOLD);
  addCard(slide, 404, 354, 504, 84, "Planner interface", "Shield executes a ranked list over 5 moves.", CMU_RED, SOFT_RED);
  setNotes(
    slide,
    slideNo,
    "Problem setup and target interface",
    [
      "Keep the math light: what matters is the interface from a continuous prediction back into a discrete coordinated planner.",
      "The repo still evaluates strict all-agents success, not only average motion quality.",
      "That is why agents-at-goal fraction becomes an important secondary diagnostic later in the deck.",
    ],
    ["main_pys/model_inputs.py", "main_pys/simulator.py", "eval_rishi_paper.py"],
  );
}

function buildMethod(slide, slideNo) {
  addSectionHeader(slide, "Method", "What we built on barath/flow-gnn", "This branch adds training, eval, and rollout code on top of the simulator.", slideNo);
  const y = 180;
  const widths = [180, 180, 180, 180];
  const labels = [
    ["Expert plans", "EECBS + BD."],
    ["Local graph", "Patch + graph."],
    ["Flow GNN", "CNN + SAGEConv."],
    ["Rollout", "Predict -> rank."],
  ];
  let x = 74;
  labels.forEach((item, idx) => {
    addCard(slide, x, y, widths[idx], 132, item[0], item[1], [CMU_NAVY, CMU_GREEN, CMU_RED, CMU_GOLD][idx], [SOFT_NAVY, SOFT_GREEN, SOFT_RED, SOFT_GOLD][idx]);
    if (idx < labels.length - 1) {
      addDividerArrow(slide, x + widths[idx] + 10, y + 48, 26, 30, CMU_RED);
    }
    x += widths[idx] + 42;
  });
  addBanner(slide, 70, 356, 820, 74, "Key change: predict a direction first, then map it back to planner actions.", CMU_RED, SOFT_RED);
  setNotes(
    slide,
    slideNo,
    "What was implemented on barath/flow-gnn",
    [
      "Use this slide to orient advisors to actual repo scope: this is not just one model file, but a full train/eval/analysis branch.",
      "The important architectural move is the shared encoder feeding both flow output and an auxiliary action head.",
      "That auxiliary head becomes central when we interpret the later transformer and action-head results.",
    ],
    ["main_pys/generative_model.py", "main_pys/train_flow.py", "eval_rishi_paper.py", "analysis_scripts/*"],
  );
}

function buildTraining(slide, slideNo) {
  addSectionHeader(slide, "Method", "How training and rollout work", "Train on flow. Roll out through a 5-way planner.", slideNo);
  addCard(slide, 46, 154, 286, 82, "Train on flow", "Predict target flow.", CMU_RED, SOFT_RED);
  addCard(slide, 46, 252, 286, 82, "Keep action head", "Auxiliary 5-way head.", CMU_NAVY, SOFT_NAVY);
  addCard(slide, 46, 350, 286, 82, "Planner bottleneck", "Planner still wants 5 moves.", CMU_GOLD, SOFT_GOLD);

  addShape(slide, "roundRect", 364, 154, 544, 278, WHITE, LIGHT_GRAY, 1);
  addText(slide, "Shared encoder", 388, 174, 180, 24, {
    size: MIN_TEXT,
    color: CMU_NAVY,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  addCard(slide, 388, 214, 154, 82, "Patch", "Obs, occ, BD.", CMU_NAVY, SOFT_NAVY);
  addCard(slide, 560, 214, 154, 82, "Neighbors", "Agent info.", CMU_RED, SOFT_RED);
  addCard(slide, 732, 214, 154, 82, "Flow", "2D velocity.", CMU_GOLD, SOFT_GOLD);
  addCard(slide, 560, 318, 154, 82, "Action", "5-way logits.", CMU_GREEN, SOFT_GREEN);
  setNotes(
    slide,
    slideNo,
    "Training objective and inference interface",
    [
      "This is the key technical slide: training lives in continuous velocity space, but deployment still has to collapse back into 5 planner actions.",
      "That is why the branch adds the action head, inference sweeps, and later transformer action-head experiments.",
      "When results break at high density, this slide gives the mechanism: the learned geometry must still survive a discrete ranking bottleneck.",
    ],
    ["main_pys/train_flow.py", "main_pys/generative_model.py", "main_pys/simulator.py", "eval_rishi_paper.py"],
  );
}

function buildProtocol(slide, slideNo) {
  addSectionHeader(slide, "Experiments", "Evaluation setup", "Same benchmark, same shield, same budget.", slideNo);
  addMetricCard(slide, 46, 156, 196, 108, "8", "held-out maps", null, CMU_RED);
  addMetricCard(slide, 258, 156, 196, 108, "1850", "eval cases", null, CMU_NAVY);
  addMetricCard(slide, 470, 156, 196, 108, "100-1000", "agents", null, CMU_GREEN);
  addMetricCard(slide, 682, 156, 226, 108, "Same shield", "Direct SSIL comparison", null, CMU_GOLD);
  addCard(slide, 80, 316, 800, 96, "What is held fixed", "Maps, scenario ladder, time budget, and CS-PIBT.", CMU_NAVY, SOFT_NAVY);
  addBanner(slide, 80, 426, 800, 48, "Metrics: success, agents at goal, runtime, cost per agent.", CMU_RED, SOFT_RED);
  setNotes(
    slide,
    slideNo,
    "Experimental protocol",
    [
      "Be explicit that the clean held-out number and the topology-exposed comparisons are different protocols, and the deck keeps them separate.",
      "This slide is where you reassure advisors the comparison to SSIL is fair.",
      "The later topology slide then uses the exposure experiment to argue that training-data coverage alone is not the main issue.",
    ],
    ["eval_rishi_paper.py", "evals/final_evals/slide_outputs/*", "evals/final_evals/topology_exposure/*"],
  );
}

async function buildBehavior(slide, slideNo) {
  addSectionHeader(slide, "Experiments", "Rollouts: solved vs hard case", "The motion looks good. Bottlenecks still jam.", slideNo);
  addShape(slide, "roundRect", 46, 150, 402, 236, WHITE, LIGHT_GRAY, 1.2);
  addShape(slide, "roundRect", 512, 150, 396, 236, WHITE, LIGHT_GRAY, 1.2);
  await addImage(slide, path.join(ROOT, "logs", "slide_gifs", "progress_smoke.gif"), 58, 168, 378, 194, "contain");
  await addImage(slide, path.join(ROOT, "logs", "slide_gifs", "planner_near_miss_warehouse_100_agents.gif"), 524, 168, 372, 194, "contain");
  addText(slide, "Solved case", 60, 394, 220, 30, {
    size: MIN_TEXT,
    color: CMU_GREEN,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  addText(slide, "Warehouse case", 526, 394, 240, 30, {
    size: MIN_TEXT,
    color: CMU_RED,
    bold: true,
    face: MONO_FACE,
    autoFit: null,
  });
  addBanner(slide, 80, 434, 796, 48, "Takeaway: the motion is there. Bottlenecks still jam.", CMU_NAVY, SOFT_NAVY);
  setNotes(
    slide,
    slideNo,
    "Planner behavior: solved vs hard cases",
    [
      "Use the solved case to show that the model and shield can produce clean coordinated motion.",
      "Use the hard warehouse case to foreshadow the main limitation: narrow bottlenecks still break the continuous-to-discrete interface.",
      "If the GIFs animate in slideshow mode, let them run briefly; if not, the first frame still anchors the discussion.",
    ],
    ["logs/slide_gifs/progress_smoke.gif", "logs/slide_gifs/planner_near_miss_warehouse_100_agents.gif"],
  );
}

function buildResults(slide, slideNo, summary) {
  addSectionHeader(slide, "Results", "Held-out results", "Good at low density. The gap opens in bottlenecks.", slideNo);

  const overall = summary.overall;
  addMetricCard(slide, 46, 146, 250, 104, `${overall.ours_clean.success_rate}%`, "Flow-GNN success", null, CMU_RED);
  addMetricCard(slide, 324, 146, 250, 104, `${overall.ssil.success_rate}%`, "SSIL success", null, CMU_NAVY);
  addMetricCard(slide, 602, 146, 306, 104, `${overall.ours_clean.at_goal_rate}%`, "Mean agents at goal", null, CMU_GOLD);

  const agentKeys = Object.keys(summary.by_agents.ours_clean).map((key) => Number(key));
  addLineChart(
    slide,
    46,
    274,
    862,
    170,
    "Success vs crowd size on the 8 held-out maps",
    agentKeys,
    [
      { name: "Flow-GNN clean", values: agentKeys.map((key) => summary.by_agents.ours_clean[String(key)].success_rate), color: CMU_RED },
      { name: "SSIL", values: agentKeys.map((key) => summary.by_agents.ssil[String(key)].success_rate), color: CMU_NAVY },
    ],
  );
  addBanner(slide, 80, 452, 796, 62, `Good through about 300 agents. Runtime still trails SSIL: ${overall.ours_clean.runtime}s vs ${overall.ssil.runtime}s.`, CMU_GREEN, SOFT_GREEN);

  setNotes(
    slide,
    slideNo,
    "Headline held-out results",
    [
      "Lead with the fairness point: this is the clean held-out comparison, not the topology-exposed reference.",
      "The encouraging result is that Flow-GNN is already competitive at 100 to 300 agents.",
      "The problem is high-density ranking and runtime, not that the model cannot move agents in the right direction at all.",
    ],
    ["presentation/advisor_update/generated_plots/grid_summary.json", "evals/final_evals/slide_outputs/tables/per_agent_summary.md"],
  );
}

async function buildTopology(slide, slideNo) {
  addSectionHeader(slide, "Results", "More map exposure does not fix it", "Training on held-out map types only moves the number a little.", slideNo);
  addBarChart(
    slide,
    92,
    154,
    776,
    254,
    "Overall success",
    ["Held-out", "Full exposure", "SSIL"],
    [
      {
        name: "Success",
        values: [59.73, 60.05, 66.43],
        color: CMU_RED,
      },
    ],
  );
  addBanner(slide, 108, 424, 744, 64, "Full map exposure adds only +0.32 points.", CMU_RED, SOFT_RED);
  setNotes(
    slide,
    slideNo,
    "Topology exposure does not close the gap",
    [
      "This is the cleanest argument that the remaining deficit is not just missing map families in training.",
      "There are small wins on Empty 48x48 and Random 32x32, but the overall number barely moves.",
      "That points back to planner alignment and action interface as the higher-leverage target.",
    ],
    ["evals/final_evals/topology_exposure/topology_exposure_slide_ready_summary.md", "evals/final_evals/topology_exposure/topology_exposure_slide_ready_highres.png"],
  );
}

function buildLimitations(slide, slideNo, summary) {
  addSectionHeader(slide, "Results", "Where failures come from", "Some runs nearly finish. Maze and bottleneck maps are still hard.", slideNo);
  const categories = ["Rnd32", "Warehouse", "Maze", "den312d"];
  addBarChart(
    slide,
    46,
    154,
    430,
    248,
    "Agents at goal within failed Flow-GNN runs",
    categories,
    [
      {
        name: "Flow-GNN clean",
        values: [
          summary.failure_at_goal_by_map.ours_clean["random-32-32-10"],
          summary.failure_at_goal_by_map.ours_clean["warehouse-10-20-10-2-1"],
          summary.failure_at_goal_by_map.ours_clean["maze-128-128-2"],
          summary.failure_at_goal_by_map.ours_clean.den312d,
        ],
        color: CMU_RED,
      },
    ],
  );
  addMetricCard(slide, 512, 156, 182, 104, `${summary.failure_at_goal.ours_clean}%`, "At-goal in failures", null, CMU_RED);
  addMetricCard(slide, 706, 156, 182, 104, `${summary.failure_at_goal.ssil}%`, "SSIL analogue", null, CMU_NAVY);
  addCard(slide, 512, 292, 376, 78, "Late deadlocks", "Rnd + warehouse often fail late.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 512, 386, 376, 78, "Still hard", "Maze + den still miss ranking.", CMU_GOLD, SOFT_GOLD);
  setNotes(
    slide,
    slideNo,
    "Where failures come from",
    [
      "Avoid overclaiming: warehouse and random failures are often near-complete, but maze and den maps still reflect real coordination difficulty.",
      "This is the slide that turns a disappointing strict-success gap into a constructive diagnosis.",
      "Tie it directly to the next-step agenda: richer planner interfaces rather than only more training data.",
    ],
    ["presentation/advisor_update/generated_plots/grid_summary.json", "evals/final_evals/slide_outputs/tables/per_map_summary.md"],
  );
}

async function buildContinuous(slide, slideNo) {
  addSectionHeader(slide, "Ongoing Work", "Continuous space is the next step", "Flow fits better once the policy predicts velocity directly.", slideNo);
  addCard(slide, 46, 164, 270, 94, "What already works", "Wins on empty maps.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 46, 282, 270, 94, "What does not yet", "Still trails ORCA.", CMU_RED, SOFT_RED);
  await addImage(slide, path.join(ROOT, "evals", "continuous_evals", "slide_outputs", "figures", "continuous_empty48_pilot_summary.png"), 340, 154, 568, 286, "contain");
  addBanner(slide, 80, 442, 792, 64, "This is the cleanest path from grid to continuous control.", CMU_NAVY, SOFT_NAVY);
  setNotes(
    slide,
    slideNo,
    "Continuous-space flow is the natural next step",
    [
      "This is the cleanest argument for the flow formulation itself: it transfers naturally into continuous control, where direct velocity prediction is a better fit.",
      "Be honest that ORCA still wins and obstacle maps are deferred.",
      "The success here is conceptual and directional: flow already beats the discretized continuous-action baseline on the empty-map pilot.",
    ],
    ["implementation_details.md", "evals/continuous_evals/slide_outputs/continuous_summary.md"],
  );
}

async function buildTransformer(slide, slideNo) {
  addSectionHeader(slide, "Ongoing Work", "Transformer results: useful, not enough", "The best transformer still trails Flow-GNN and SSIL.", slideNo);
  addCard(slide, 46, 164, 270, 94, "Best result", "42.1% with action head.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 46, 282, 270, 94, "Lesson", "Action labels help.", CMU_RED, SOFT_RED);
  await addImage(slide, path.join(ROOT, "evals", "transformer", "slide_outputs", "full_comparison", "figures", "shared_case_overall_comparison_clean_labels.png"), 340, 154, 568, 286, "contain");
  addBanner(slide, 80, 442, 792, 64, "Same lesson: action structure still matters.", CMU_NAVY, SOFT_NAVY);
  setNotes(
    slide,
    slideNo,
    "Transformer action selection is informative, but not the fix",
    [
      "Use this as a negative but useful result: more expressive sequence models did not automatically solve the planner-interface problem.",
      "The action-head transformer is clearly the best transformer variant, which again points to action alignment.",
      "That keeps the future-work focus disciplined rather than chasing architecture alone.",
    ],
    ["evals/transformer/slide_outputs/transformer_summary.md", "evals/transformer/slide_outputs/full_comparison/shared_case_comparison_summary.md"],
  );
}

function buildFuture(slide, slideNo) {
  addSectionHeader(slide, "Future Work", "Future work: a lattice action set", "Change the planner interface, not just the model.", slideNo);
  addCard(slide, 50, 164, 240, 94, "Current bottleneck", "Flow collapses into 5 actions.", CMU_RED, SOFT_RED);
  addDividerArrow(slide, 310, 196, 58, 30, CMU_RED);
  addCard(slide, 386, 164, 200, 94, "Lattice", "Short motion primitives.", CMU_NAVY, SOFT_NAVY);
  addDividerArrow(slide, 606, 196, 58, 30, CMU_RED);
  addCard(slide, 682, 164, 226, 94, "Planner payoff", "Richer branches to rank.", CMU_GREEN, SOFT_GREEN);

  addCard(slide, 92, 308, 776, 94, "Why this fits", "Flow already points in a direction, so lattice is a clean next step.", CMU_GOLD, SOFT_GOLD);
  addBanner(slide, 92, 416, 776, 64, "Keep the shield. Stop forcing every choice through 5 moves.", CMU_RED, SOFT_RED);
  setNotes(
    slide,
    slideNo,
    "Future work: replace cardinal actions with a lattice interface",
    [
      "This is the main future-work thesis for advisors: the next grid-world experiment should change the action interface, not only retrain the same 5-way bottleneck harder.",
      "A lattice interface is also a good bridge conceptually between the grid-world and continuous-space branches.",
      "Frame this as the most promising route to turn the current diagnosis into an actionable grid-world experiment.",
    ],
    ["PAPER_PLAN.md", "main_pys/simulator.py", "FLOW_VS_SSIL_ABLATION_PLAN.md"],
  );
}

function buildTakeaways(slide, slideNo) {
  addSectionHeader(slide, "Close", "Takeaways", "Flow looks promising, but the 5-way interface is still the bottleneck.", slideNo);
  addCard(slide, 44, 166, 270, 102, "1. Promise", "Competitive at lower density.", CMU_GREEN, SOFT_GREEN);
  addCard(slide, 344, 166, 270, 102, "2. Bottleneck", "The 5-way interface is still the limiter.", CMU_RED, SOFT_RED);
  addCard(slide, 644, 166, 270, 102, "3. Next step", "Lattice is the clearest grid follow-up.", CMU_NAVY, SOFT_NAVY);
  addCard(slide, 94, 322, 772, 84, "Advisor question", "Next sprint: lattice, or one more action-head pass?", CMU_GOLD, SOFT_GOLD);
  setNotes(
    slide,
    slideNo,
    "Takeaways and advisor discussion",
    [
      "Close by making the diagnosis actionable rather than defensive.",
      "The positive story is strong enough to justify the branch: flow is not a dead end, it is revealing where the interface really matters.",
      "End by asking for guidance on whether to pursue the lattice interface immediately or do one more action-alignment round first.",
    ],
    ["PAPER_PLAN.md", "FLOW_VS_SSIL_ABLATION_PLAN.md", "evals/final_evals/slide_outputs/takeaways.md"],
  );
}

async function buildBackup12Map(slide, slideNo) {
  addSectionHeader(slide, "Backup", "12-map comparison", "Backup: broader 12-map view.", slideNo);
  await addImage(slide, path.join(ROOT, "evals", "final_evals", "rishi12_ours_vs_rishi_1v1", "rishi12_ours_vs_ssil_summary_2row_highres.png"), 40, 126, 878, 340, "contain");
  setNotes(
    slide,
    slideNo,
    "12-map panel comparison",
    [
      "Use only if the discussion shifts toward topology familiarity or the broader 12-map panel.",
      "The main point remains the same: the gap persists and is not obviously erased by more familiar map families.",
    ],
    ["evals/final_evals/rishi12_ours_vs_rishi_1v1/rishi12_ours_vs_ssil_highres_2x6_report.md"],
  );
}

async function buildBackupTransformer(slide, slideNo) {
  addSectionHeader(slide, "Backup", "Transformer scaling", "Backup: full transformer scaling curve.", slideNo);
  await addImage(slide, path.join(ROOT, "evals", "transformer", "slide_outputs", "full_comparison", "figures", "shared_case_success_by_agents.png"), 42, 122, 876, 348, "contain");
  setNotes(
    slide,
    slideNo,
    "Transformer detailed scaling",
    [
      "Keep this as a technical backup if advisors ask whether the transformer effort is competitive already.",
      "The answer in the repo today is no: useful diagnostic, not yet a headline model.",
    ],
    ["evals/transformer/slide_outputs/full_comparison/figures/shared_case_success_by_agents.png"],
  );
}

async function writeSpeakerNotes() {
  const lines = ["# Speaker Notes", ""];
  for (const item of notes.sort((a, b) => a.slideNo - b.slideNo)) {
    lines.push(`## Slide ${item.slideNo}: ${item.title}`);
    lines.push("");
    item.bullets.forEach((bullet) => lines.push(`- ${bullet}`));
    if (item.sources.length) {
      lines.push("");
      lines.push("Sources:");
      item.sources.forEach((src) => lines.push(`- ${src}`));
    }
    lines.push("");
  }
  await fs.writeFile(SPEAKER_NOTES_PATH, `${lines.join("\n")}\n`, "utf8");
}

async function renderPreviews(presentation) {
  await fs.rm(PREVIEW_DIR, { recursive: true, force: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const previewIndices = Array.from({ length: presentation.slides.count }, (_, index) => index);
  for (const index of previewIndices) {
    const slide = presentation.slides.getItem(index);
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    const bytes = await png.bytes();
    await fs.writeFile(path.join(PREVIEW_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`), Buffer.from(bytes));
  }
}

async function main() {
  const templateBlob = await FileBlob.load(TEMPLATE_PATH);
  const presentation = await PresentationFile.importPptx(templateBlob);
  const summary = JSON.parse(await fs.readFile(DATA_PATH, "utf8"));

  const titleSlide = presentation.slides.getItem(0);
  const blankBase = presentation.slides.getItem(1);

  const contentSlides = [blankBase];
  let lastBlank = blankBase;
  for (let i = 1; i < 16; i += 1) {
    lastBlank = lastBlank.duplicate();
    contentSlides.push(lastBlank);
  }

  buildTitleSlide(titleSlide);
  await buildMotivation(contentSlides[0], 2);
  buildRelated(contentSlides[1], 3);
  buildProblem(contentSlides[2], 4);
  buildMethod(contentSlides[3], 5);
  buildTraining(contentSlides[4], 6);
  buildProtocol(contentSlides[5], 7);
  await buildBehavior(contentSlides[6], 8);
  buildResults(contentSlides[7], 9, summary);
  await buildTopology(contentSlides[8], 10);
  buildLimitations(contentSlides[9], 11, summary);
  await buildContinuous(contentSlides[10], 12);
  await buildTransformer(contentSlides[11], 13);
  buildFuture(contentSlides[12], 14);
  buildTakeaways(contentSlides[13], 15);
  await buildBackup12Map(contentSlides[14], 16);
  await buildBackupTransformer(contentSlides[15], 17);

  await fs.mkdir(OUT_DIR, { recursive: true });
  await writeSpeakerNotes();

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT_PPTX);
  await renderPreviews(presentation);

  console.log(`Wrote ${OUTPUT_PPTX}`);
  console.log(`Wrote ${SPEAKER_NOTES_PATH}`);
  console.log(`Rendered previews into ${PREVIEW_DIR}`);
}

await main();
