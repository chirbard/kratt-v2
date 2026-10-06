"""Shrinks the path data in an SVG without changing the picture.

Coordinates become relative and rounded to 0.1 px, axis-aligned lines use H/V,
points lying on a straight line are dropped, and a command letter is written
only when it changes. Used by charts.py and modules.py.
"""
import re

TOKEN = re.compile(r"[MLHVCSQTAZmlhvcsqtaz]|-?\d*\.?\d+(?:e-?\d+)?")
ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def number(value):
    text = f"{round(value, 1):g}"
    return "0" if text == "-0" else text


def join(values):
    out = ""
    for value in values:
        text = number(value)
        out += text if not out or text.startswith("-") else " " + text
    return out


def absolute_segments(d):
    """(command, absolute points) for every segment; handles relative input."""
    tokens = TOKEN.findall(d)
    segments, x, y, start, index, command = [], 0.0, 0.0, (0.0, 0.0), 0, None
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command in "Zz":
                segments.append(("Z", []))
                x, y = start
                continue
        upper, relative = command.upper(), command.islower()
        values = [float(token) for token in tokens[index:index + ARITY[upper]]]
        index += ARITY[upper]
        if upper == "H":
            x = values[0] + (x if relative else 0)
            segments.append(("L", [x, y]))
        elif upper == "V":
            y = values[0] + (y if relative else 0)
            segments.append(("L", [x, y]))
        elif upper == "A":
            end = [values[5] + (x if relative else 0), values[6] + (y if relative else 0)]
            segments.append(("A", values[:5] + end))
            x, y = end
        else:
            points = []
            for offset in range(0, len(values), 2):
                points += [values[offset] + (x if relative else 0), values[offset + 1] + (y if relative else 0)]
            segments.append((upper, points))
            x, y = points[-2], points[-1]
            if upper == "M":
                start = (x, y)
                command = "l" if relative else "L"
    return segments


def drop_collinear(segments):
    out = []
    for segment in segments:
        if segment[0] == "L" and len(out) >= 2 and out[-1][0] == "L" and out[-2][0] in "ML":
            ax, ay = out[-2][1][-2:]
            bx, by = out[-1][1]
            cx, cy = segment[1]
            if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) < 0.05:
                out[-1] = segment
                continue
        out.append(segment)
    return out


def compact(d):
    out, last, x, y, start = "", "", 0.0, 0.0, (0.0, 0.0)
    for command, values in drop_collinear(absolute_segments(d)):
        if command == "Z":
            out += "z"
            last, (x, y) = "z", start
            continue
        if command == "A":
            letter, numbers = "a", values[:5] + [values[5] - x, values[6] - y]
        elif command == "L" and round(values[1] - y, 1) == 0:
            letter, numbers = "h", [values[0] - x]
        elif command == "L" and round(values[0] - x, 1) == 0:
            letter, numbers = "v", [values[1] - y]
        else:
            letter = command.lower()
            numbers = [value - (x if index % 2 == 0 else y) for index, value in enumerate(values)]
        text = join(numbers)
        out += (text if text.startswith("-") else " " + text) if letter == last and letter != "m" else letter + text
        last = "l" if letter == "m" else letter
        x, y = values[-2], values[-1]
        if command == "M":
            start = (x, y)
    return out


def shrink(svg):
    return re.sub(r'(?<=\sd=")[^"]*', lambda match: compact(match.group(0)), svg)
