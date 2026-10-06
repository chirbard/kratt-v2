"""Draws the README charts from this repository's own history.

    uv run --with matplotlib docs/charts.py

1. code-size: lines of Rust per subsystem at every commit on main.
2. predict-f1: the change-impact eval (numbers from its curve.csv, July 2026).
Each in a light and a dark version, for GitHub's <picture> theme switch.
"""
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# SVG with real <text> in a generic sans-serif -- a font every system has, so
# GitHub's no-web-fonts sandbox is no problem -- and no timestamp, so a redraw
# of unchanged data is byte-identical.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["svg.hashsalt"] = "kratt-v2"
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from svgmin import shrink

OUT = Path(__file__).parent

# Order is the stacking order, bottom to top; colours are the validated
# categorical slots 1-7 in that order.
GROUPS = [
    ("storage & retrieval", ("src/memory/db", "src/graph", "src/git", "src/memory/git", "src/memory/embed", "src/memory/graph", "src/memory/toon", "src/memory/retrieval")),
    ("extraction & prompts", ("src/memory/agent", "src/memory_agent", "src/memory_prompt", "src/memory/ingest", "src/prompts", "src/memory/extract")),
    ("autonomous mind", ("src/memory/dream", "src/decay", "src/memory/synthesis", "src/memory/explore", "src/memory/news",
                         "src/memory/scheduler", "src/memory/reinforce", "src/memory/decay", "src/memory/hygiene",
                         "src/memory/heavy", "src/memory/strategy", "src/memory/routing", "src/memory/ledger",
                         "src/memory/sequence", "src/memory/task_chain", "src/memory/loops", "src/memory/interest",
                         "src/memory/autonomy")),
    ("interfaces: TUI & Telegram", ("src/tui/", "src/telegram/", "src/bin/")),
    ("LLM & web APIs, tracing, glue", ("src/api/", "src/web", "src/trace", "src/main", "src/runlog", "src/paths",
                                       "src/timeutil", "src/lib")),
    ("eval harness", ("src/eval/",)),
    ("conversation: turns & sessions", ("src/turn", "src/session", "src/seat_replay", "src/chat", "src/history")),
]
OTHER = "LLM & web APIs, tracing, glue"

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
         "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]}
DARK = {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]}


REPO = Path(__file__).resolve().parent.parent


def save_svg(figure, path, theme):
    figure.savefig(path, facecolor=theme["surface"], metadata={"Date": None})
    path.write_text(shrink(path.read_text()))


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def group_of(path):
    for name, prefixes in GROUPS:
        for prefix in prefixes:
            if path.startswith(prefix):
                return name
    return OTHER


def sizes_at(commit):
    totals = {name: 0 for name, _ in GROUPS}
    listing = git("ls-tree", "-r", "-l", commit, "src")
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        if not path.endswith(".rs"):
            continue
        blob = meta.split()[2]
        lines = git("cat-file", "-p", blob).count("\n")
        totals[group_of(path)] += lines
    return totals


def history():
    rows = []
    # Only commits that touched src/: the archive commits added in October
    # would otherwise stretch the timeline months past the work.
    for line in git("log", "--first-parent", "--reverse", "--format=%H %cI", "main", "--", "src").splitlines():
        commit, when = line.split()
        rows.append((datetime.fromisoformat(when), sizes_at(commit)))
    return rows


def style(axis, theme):
    axis.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(theme["muted"])
    axis.tick_params(colors=theme["muted"], labelsize=9, length=0)
    axis.grid(axis="y", color=theme["grid"], linewidth=0.8)
    axis.set_axisbelow(True)


def code_size(rows, tag_date, theme, path):
    figure, axis = plt.subplots(figsize=(10, 4.8), dpi=150, facecolor=theme["surface"])
    style(axis, theme)
    dates = [row[0] for row in rows]
    stacks = [[row[1][name] for row in rows] for name, _ in GROUPS]
    axis.stackplot(dates, *stacks, colors=theme["series"], edgecolor=theme["surface"], linewidth=0.6)
    axis.axvline(tag_date, color=theme["ink"], linewidth=1, linestyle=(0, (3, 3)))
    peak = max(sum(row[1].values()) for row in rows)
    # A fixed offset, not leading spaces: the viewer's sans-serif sets the width.
    axis.annotate("v2.0 tag — then the slim-down", (tag_date, peak * 1.04), xytext=(6, 0),
                  textcoords="offset points", color=theme["ink"], fontsize=9, va="bottom")
    axis.set_ylim(0, peak * 1.15)
    axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{int(value / 1000)}k"))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.set_title("kratt-v2: lines of Rust per subsystem, every commit", loc="left", color=theme["ink"], fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in theme["series"]]
    legend = axis.legend(handles, [name for name, _ in GROUPS], loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    figure.tight_layout()
    save_svg(figure, path, theme)
    plt.close(figure)


# From kratt-data-dev/predict-eval/curve.csv (run 2026-07-24): F1 over file
# paths when predicting which files a past commit touched, after N training
# commits. "cold" sees only the task and the file tree; "memory" also gets
# kratt's graph built from the training commits.
TRAINING = [0, 10, 20, 40]
F1 = {"without memory (task + file tree)": [0.322, 0.259, 0.323, 0.274],
      "with kratt's memory": [0.353, 0.248, 0.231, 0.332]}


def predict_f1(theme, path):
    figure, axis = plt.subplots(figsize=(7, 4), dpi=150, facecolor=theme["surface"])
    style(axis, theme)
    for index, (name, values) in enumerate(F1.items()):
        color = theme["series"][index]
        axis.plot(TRAINING, values, color=color, linewidth=2, marker="o", markersize=7,
                  markeredgecolor=theme["surface"], markeredgewidth=2)
        axis.annotate(name, (TRAINING[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                      color=theme["ink"], fontsize=9, va="center")
    axis.set_xticks(TRAINING)
    # Room on the right for the line labels in any reader's sans-serif.
    axis.set_xlim(-2, 75)
    axis.set_ylim(0, 0.5)
    axis.set_xlabel("training commits the memory had learned from", color=theme["muted"], fontsize=9)
    axis.set_ylabel("F1, predicted vs actual files", color=theme["muted"], fontsize=9)
    axis.set_title("More experience did not help", loc="left", color=theme["ink"], fontsize=12)
    figure.tight_layout()
    save_svg(figure, path, theme)
    plt.close(figure)


def main():
    rows = history()
    tag_date = datetime.fromisoformat(git("log", "-1", "--format=%cI", "v2.0").strip())
    peak = max(rows, key=lambda row: sum(row[1].values()))
    print(f"{len(rows)} commits; peak {sum(peak[1].values())} lines on {peak[0]:%Y-%m-%d}; final {sum(rows[-1][1].values())}")
    for mode, theme in (("light", LIGHT), ("dark", DARK)):
        code_size(rows, tag_date, theme, OUT / f"code-size-{mode}.svg")
        predict_f1(theme, OUT / f"predict-f1-{mode}.svg")


if __name__ == "__main__":
    main()
