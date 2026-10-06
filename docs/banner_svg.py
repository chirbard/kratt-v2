"""Builds docs/banner.svg: the banner from docs/banner.html as a pure vector file.

GitHub shows README images in a sandbox that blocks web fonts, so the text is
converted to outlines -- no font is needed to display it. Every run is placed
where Chrome put it (docs/banner-layout.json, measured from banner.html) and
shaped with HarfBuzz, so kerning and spacing match the browser. Glyphs used
more than once are defined once and reused.

    uv run --with fonttools --with uharfbuzz docs/banner_svg.py <fonts-dir>

<fonts-dir> holds, from github.com/google/fonts (OFL): InstrumentSerif-Italic.ttf,
Inter[opsz,wght].ttf, JetBrainsMono[wght].ttf.
"""
import json
import sys
from pathlib import Path

import uharfbuzz as hb

HERE = Path(__file__).parent
W, H = 1600, 640
BG, FG, MUTED, MUTED_2, ACCENT = "#0a0a0c", "#ece8e1", "#6b6b70", "#9a9a9f", "#ff5e1a"


class Face:
    def __init__(self, path, variations):
        blob = hb.Blob.from_file_path(str(path))
        self.font = hb.Font(hb.Face(blob))
        if variations:
            self.font.set_variations(variations)
        self.upem = self.font.face.upem
        self.key = path.stem.split("[")[0] + "".join(f"-{k}{v}" for k, v in sorted(variations.items()))


class PathPen:
    """Collects one glyph outline in font units, y up."""

    def __init__(self):
        self.parts = []

    def move_to(self, x, y):
        self.parts.append(("M", (x, y)))

    def line_to(self, x, y):
        self.parts.append(("L", (x, y)))

    def quadratic_to(self, c1x, c1y, x, y):
        self.parts.append(("Q", (c1x, c1y, x, y)))

    def cubic_to(self, c1x, c1y, c2x, c2y, x, y):
        self.parts.append(("C", (c1x, c1y, c2x, c2y, x, y)))

    def close_path(self):
        self.parts.append(("Z", ()))


def glyph_path(face, gid):
    pen = PathPen()
    face.font.draw_glyph(gid, _funcs(pen), None)
    return compact(pen.parts)


def number(value):
    text = f"{round(value, 1):g}"
    return "0" if text == "-0" else text


def join_numbers(values):
    """SVG lets a minus sign stand in for the separator: 3-4 means 3 -4."""
    out = ""
    for value in values:
        text = number(value)
        out += text if not out or text.startswith("-") else " " + text
    return out


# Relative commands, H and V where a line is axis-aligned, and a command letter
# only when it changes -- about half the size of plain absolute paths.
def compact(parts):
    out, last, x, y, start = "", "", 0.0, 0.0, (0.0, 0.0)
    for command, values in parts:
        flipped = [value if index % 2 == 0 else -value for index, value in enumerate(values)]
        if command == "Z":
            out += "z"
            last, (x, y) = "z", start
            continue
        if command == "M":
            letter, numbers = "m", [flipped[0] - x, flipped[1] - y]
            start = (flipped[0], flipped[1])
        elif command == "L" and round(flipped[1] - y, 1) == 0:
            letter, numbers = "h", [flipped[0] - x]
        elif command == "L" and round(flipped[0] - x, 1) == 0:
            letter, numbers = "v", [flipped[1] - y]
        else:
            letter = command.lower()
            numbers = [value - (x if index % 2 == 0 else y) for index, value in enumerate(flipped)]
        repeat = letter == last and letter != "m"
        text = join_numbers(numbers)
        out += (" " + text if not text.startswith("-") else text) if repeat else letter + text
        last = "l" if letter == "m" else letter
        x, y = flipped[-2], flipped[-1]
    return out


def _funcs(pen):
    funcs = hb.DrawFuncs()
    funcs.set_move_to_func(lambda x, y, _: pen.move_to(x, y))
    funcs.set_line_to_func(lambda x, y, _: pen.line_to(x, y))
    funcs.set_quadratic_to_func(lambda a, b, x, y, _: pen.quadratic_to(a, b, x, y))
    funcs.set_cubic_to_func(lambda a, b, c, d, x, y, _: pen.cubic_to(a, b, c, d, x, y))
    funcs.set_close_path_func(lambda _: pen.close_path())
    return funcs


class Builder:
    def __init__(self):
        self.defs = {}
        self.uses = []

    def run(self, face, text, x, baseline, size, colour, letter_spacing=0.0, features=None):
        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        hb.shape(face.font, buffer, features or {})
        scale = size / face.upem
        pen_x = x
        glyphs = []
        for info, position in zip(buffer.glyph_infos, buffer.glyph_positions):
            glyph_id = f"{face.key}-{info.codepoint}"
            if glyph_id not in self.defs:
                self.defs[glyph_id] = glyph_path(face, info.codepoint)
            gx = pen_x + position.x_offset * scale
            gy = baseline - position.y_offset * scale
            if self.defs[glyph_id]:
                glyphs.append((glyph_id, gx, gy))
            pen_x += position.x_advance * scale + letter_spacing
        self.uses.append((colour, scale, glyphs))
        return pen_x

    def svg(self, background, glow):
        ids = {glyph_id: f"g{index}" for index, glyph_id in enumerate(self.defs)}
        used = set()
        body = []
        for colour, scale, glyphs in self.uses:
            body.append(f'<g fill="{colour}" transform="scale({scale:.6g})">')
            for glyph_id, gx, gy in glyphs:
                used.add(glyph_id)
                body.append(f'<use href="#{ids[glyph_id]}" x="{number(gx / scale)}" y="{number(gy / scale)}"/>')
            body.append("</g>")
        defs = "".join(f'<path id="{ids[glyph_id]}" d="{self.defs[glyph_id]}"/>' for glyph_id in self.defs if glyph_id in used)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"'
                f' aria-label="n = 1. Learn from one game, not a million. An attempt to build a learner like a person:'
                f' imperfect, but sample-efficient.">'
                f"<defs>{defs}{glow[0]}</defs>{background}{glow[1]}{''.join(body)}</svg>")


def background():
    """The same seeded grid, glow and points as the canvas in banner.html."""
    grid = "".join(f"M{x + 0.5} 0v{H}" for x in range(0, W, 80)) + "".join(f"M0 {y + 0.5}h{W}" for y in range(0, H, 80))
    seed = 7

    def random():
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return (seed - 1) / 2147483646

    points = []
    for _ in range(85):
        points.append((random() * W, random() * H, random() * 1.6 + 0.4, random() * 0.4 + 0.12))
    # Opacities are rounded to a few levels so each level is one path: at 1-6%
    # (lines) and 12-52% (points) the rounding is invisible, the bytes are not.
    lines = {}
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            dx, dy = points[first][0] - points[second][0], points[first][1] - points[second][1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < 120:
                level = round(0.06 * (1 - distance / 120), 2)
                lines.setdefault(level, []).append(
                    f"M{number(points[first][0])} {number(points[first][1])}L{number(points[second][0])} {number(points[second][1])}")
    dots = {}
    for x, y, r, a in points:
        r = round(r, 1)
        dots.setdefault(round(a * 20) / 20, []).append(f"M{number(x - r)} {number(y)}a{number(r)} {number(r)} 0 1 0 {number(2 * r)} 0a{number(r)} {number(r)} 0 1 0 {number(-2 * r)} 0")
    line_paths = "".join(f'<path d="{"".join(parts)}" stroke-opacity="{level:g}"/>' for level, parts in sorted(lines.items()) if level > 0)
    dot_paths = "".join(f'<path d="{"".join(parts)}" fill-opacity="{level:g}"/>' for level, parts in sorted(dots.items()))
    return (f'<rect width="{W}" height="{H}" fill="{BG}"/>'
            f'<path d="{grid}" stroke="{FG}" stroke-opacity="0.03"/>'
            f'<radialGradient id="warm" gradientUnits="userSpaceOnUse" cx="{W / 2}" cy="{H / 2}" r="{W * 0.55}">'
            f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.07"/><stop offset="0.5" stop-color="{ACCENT}" stop-opacity="0.015"/>'
            f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/></radialGradient>'
            f'<rect width="{W}" height="{H}" fill="url(#warm)"/>'
            f'<g stroke="{FG}">{line_paths}</g><g fill="{FG}">{dot_paths}</g>')


def glow(box):
    """The .one::before glow: a radial ellipse in a box under the 1, blurred 30px."""
    height = box["bottom"] - box["top"]
    width = box["right"] - box["left"]
    x0, x1 = box["left"] - 0.2 * width, box["right"] + 0.2 * width
    y1 = box["bottom"] - 0.12 * height
    y0 = y1 - 0.3 * height
    defs = ('<radialGradient id="glowfill" r="0.7071"><stop offset="0" stop-color="#ff5e1a" stop-opacity="0.45"/>'
            '<stop offset="0.7" stop-color="#ff5e1a" stop-opacity="0"/></radialGradient>'
            '<filter id="blur" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="30"/></filter>')
    shape = f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" fill="url(#glowfill)" filter="url(#blur)"/>'
    return defs, shape


def main():
    fonts = Path(sys.argv[1])
    layout = json.loads((HERE / "banner-layout.json").read_text())
    serif = Face(fonts / "InstrumentSerif-Italic.ttf", {})
    # Google Fonts serves Inter at its default optical size (14), not the font
    # size: opsz 14 reproduces Chrome's measured run width to 0.01 px.
    inter = Face(fonts / "Inter[opsz,wght].ttf", {"wght": 300, "opsz": 14})
    mono = Face(fonts / "JetBrainsMono[wght].ttf", {"wght": 400})
    builder = Builder()
    for key, text in (("tl1", "KRATT-V2"), ("tl2", "RESEARCH RECORD"), ("tr1", "JUNE – JULY 2026"), ("tr2", "ARCHIVED")):
        spot = layout[key]
        builder.run(mono, text, spot["left"], spot["baseline"], 13, MUTED, letter_spacing=0.18 * 13, features={"liga": False, "calt": False})
    for key, text, colour in (("fn", "n", MUTED_2), ("fe", "=", MUTED_2), ("f1", "1", ACCENT)):
        spot = layout[key]
        builder.run(serif, text, spot["left"], spot["baseline"], spot["fontSize"], colour, letter_spacing=-0.04 * spot["fontSize"])
    builder.run(serif, "Learn from one game, not a million.", layout["tg"]["left"], layout["tg"]["baseline"], 44, FG, letter_spacing=-0.44)
    sub = layout["s1"]
    builder.run(inter, "An attempt to build a learner like a person — imperfect, but ", sub["left"], sub["baseline"], 19, MUTED_2)
    builder.run(serif, "sample-efficient", layout["s2"]["left"], layout["s2"]["baseline"], layout["s2"]["fontSize"], FG)
    builder.run(inter, ".", layout["s3"]["left"], layout["s3"]["baseline"], 19, MUTED_2)
    out = HERE / "banner.svg"
    out.write_text(builder.svg(background(), glow(layout["f1"])))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(builder.defs)} glyph outlines)")


if __name__ == "__main__":
    main()
