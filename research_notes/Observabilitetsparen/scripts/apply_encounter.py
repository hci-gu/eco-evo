"""Write the encounter components (encounter_components.py library listing) into
a copy of fg_library.yaml: in each listed `<obs>_observes_<target>` block the
`neighbour_weight` line is replaced by `encounter_own` / `encounter_neighbour`
(flow lists); every other line, comment and the BOM stay as they are.

Run: python3 -I apply_encounter.py <listing.txt> <library_in> <library_out>
"""
import re
import sys


def main(listing, src, dst):
    table = {}
    for line in open(listing, encoding="utf-8"):
        m = re.match(r"(\S+) encounter_own: (\[.*\]) encounter_neighbour: (\[.*\])$", line.strip())
        if m:
            table[m.group(1)] = (m.group(2), m.group(3))
    raw = open(src, "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").split("\n")
    out, key, done = [], None, set()
    head = re.compile(r"^  (\S+_observes_\S+):\s*$")
    for line in lines:
        m = head.match(line)
        if m:
            key = m.group(1)
        elif not line.startswith("    "):
            key = None
        if key in table and line.strip().startswith(("neighbour_weight:", "encounter_own:",
                                                      "encounter_neighbour:")):
            continue
        out.append(line)
        if key in table and line.strip() == "observes: true":
            own, nbr = table[key]
            out.append(f"    encounter_own: {own}")
            out.append(f"    encounter_neighbour: {nbr}")
            done.add(key)
    missing = sorted(set(table) - done)
    if missing:
        raise SystemExit(f"not written (no 'observes: true' line): {missing}")
    text = "\n".join(out)
    open(dst, "wb").write((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    print(f"{len(done)} pairs written to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
