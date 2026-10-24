"""Parse an ICES Standard Graphs ViewSourceData HTML page (untrusted input) to CSV on stdout.
Usage: python3 -I sg_source_to_csv.py page.html"""
import html, re, sys

t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
rows = re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S)
out = []
for r in rows:
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
    if cells and (cells[0] == "Year" or re.fullmatch(r"\d{4}", cells[0] or "")):
        out.append(cells)
for c in out:
    print(",".join(x.replace(",", "") for x in c))
