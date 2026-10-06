"""Draws the module-dependency charts for the README.

The inputs in docs/modules/ come from cargo-modules (0.26), with use edges:

    git checkout v2.0 && cargo modules dependencies --bin kratt-v2 \\
        --no-externs --no-fns --no-sysroot --no-traits --no-types > docs/modules/v2.0.dot
    git checkout main && cargo modules dependencies \\
        --no-externs --no-fns --no-sysroot --no-traits --no-types > docs/modules/slimmed.dot

then:  python3 docs/modules.py   (needs Graphviz)

Drawn as the ownership tree (which module contains which), coloured by the
subsystems of docs/charts.py, with a legend. Of the use edges only those that
break the rule in CLAUDE.md -- core knows nothing of the interfaces, the
interfaces know nothing of memory internals -- are drawn, in red; all of them
are counted in the title.
"""
import re
import subprocess
from pathlib import Path

from charts import DARK, GROUPS, LIGHT, group_of
from svgmin import shrink

HERE = Path(__file__).parent
CRATE = "kratt_v2"
INTERFACES = ("tui", "telegram")
CORE = ("memory", "api")
INTERNALS = "memory::db"
EDGE = re.compile(r'^\s*"([^"]+)" -> "([^"]+)" \[label="(owns|uses)"')
NODE = re.compile(r'^\s*"([^"]+)" \[label=')
VIOLATION = {"light": "#d03b3b", "dark": "#ff6b6b"}


def load(path):
    modules, owns, uses = set(), set(), set()
    for line in path.read_text().splitlines():
        node = NODE.match(line)
        if node:
            modules.add(node.group(1))
        edge = EDGE.match(line)
        if edge and edge.group(1) != edge.group(2):
            (owns if edge.group(3) == "owns" else uses).add((edge.group(1), edge.group(2)))
    return modules, owns, uses


def local(module):
    return module[len(CRATE) + 2:] if module.startswith(CRATE + "::") else ""


def breaks_boundary(source, target):
    source, target = local(source), local(target)
    core_to_interface = source.startswith(CORE) and target.startswith(INTERFACES)
    interface_to_internals = source.startswith(INTERFACES) and target.startswith(INTERNALS)
    return core_to_interface or interface_to_internals


def colour(module, theme):
    path = "src/" + local(module).replace("::", "/") if local(module) else "src/main"
    # A parent module ("tui") is classified like its children ("tui/app").
    names = [name for name, _ in GROUPS]
    return theme["series"][names.index(group_of(path + "/"))]


# Dark text on the light fills (yellow, aqua, pink), white on the rest, so
# every label clears contrast.
def text_on(fill):
    red, green, blue = int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
    return "#0b0b0b" if 0.299 * red + 0.587 * green + 0.114 * blue > 140 else "#ffffff"


# The ownership tree is the skeleton: it reads like the source folders. Of the
# use edges only the rule-breaking ones are drawn -- all 257 at once is a
# hairball that says nothing.
def dot_source(modules, owns, broken, theme, mode, title):
    lines = [
        "digraph modules {",
        f'  graph [bgcolor="{theme["surface"]}", rankdir=LR, nodesep=0.12, ranksep=0.5, pad=0.4,'
        f' fontname="Helvetica", labelloc=t, label=<{legend(theme, mode, title)}>];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=0,'
        ' height=0.32, margin="0.12,0.04"];',
        f'  edge [color="{theme["muted"]}", arrowhead=none, penwidth=1];',
    ]
    for module in sorted(modules):
        fill = colour(module, theme)
        name = local(module).split("::")[-1] if local(module) else "kratt-v2"
        lines.append(f'  "{module}" [label="{name}", fillcolor="{fill}", fontcolor="{text_on(fill)}"];')
    for parent, child in sorted(owns):
        lines.append(f'  "{parent}" -> "{child}";')
    for source, target in sorted(broken):
        lines.append(f'  "{source}" -> "{target}" [color="{VIOLATION[mode]}", penwidth=2.5, arrowhead=normal,'
                     f' arrowsize=0.9, constraint=false];')
    lines.append("}")
    return "\n".join(lines)


def legend(theme, mode, title):
    cells = ""
    for index, (name, _) in enumerate(GROUPS):
        cells += (f'<TD BGCOLOR="{theme["series"][index]}" WIDTH="14" HEIGHT="14"></TD>'
                  f'<TD ALIGN="LEFT"><FONT COLOR="{theme["ink"]}" POINT-SIZE="12"> {name.replace("&", "&amp;")}   </FONT></TD>')
    cells += (f'<TD><FONT COLOR="{VIOLATION[mode]}" POINT-SIZE="16">&#10230;</FONT></TD>'
              f'<TD ALIGN="LEFT"><FONT COLOR="{theme["ink"]}" POINT-SIZE="12"> breaks the boundary rule</FONT></TD>')
    return (f'<TABLE BORDER="0" CELLSPACING="4">'
            f'<TR><TD COLSPAN="16" ALIGN="LEFT"><FONT COLOR="{theme["ink"]}" POINT-SIZE="20">{title}</FONT></TD></TR>'
            f'<TR><TD COLSPAN="16" ALIGN="LEFT"><FONT COLOR="{theme["muted"]}" POINT-SIZE="12">'
            f'grey lines: which module contains which (the source tree)</FONT></TD></TR>'
            f'<TR>{cells}</TR></TABLE>')


def main():
    for name, title in (("v2.0", "kratt-v2 at v2.0"), ("slimmed", "kratt-v2 after the slim-down")):
        modules, owns, uses = load(HERE / "modules" / f"{name}.dot")
        broken = [edge for edge in uses if breaks_boundary(*edge)]
        print(f"{name}: {len(modules)} modules, {len(uses)} dependencies, {len(broken)} boundary violations")
        full_title = (f"{title}: {len(modules)} modules, {len(uses)} dependencies between them,"
                      f" {len(broken)} break the boundary rule")
        for mode, theme in (("light", LIGHT), ("dark", DARK)):
            source = dot_source(modules, owns, broken, theme, mode, full_title)
            out = HERE / f"modules-{name}-{mode}.svg"
            drawn = subprocess.run(["dot", "-Tsvg"], input=source, text=True, check=True, capture_output=True).stdout
            out.write_text(shrink(drawn))


if __name__ == "__main__":
    main()
