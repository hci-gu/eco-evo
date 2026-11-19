"""Strip an HTML article page to plain text (one block per paragraph).

Usage: python3 -I html2txt.py in.html out.txt
"""
import html.parser
import re
import sys


class P(html.parser.HTMLParser):
    BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "li", "tr", "table", "section", "br", "caption", "figcaption"}
    SKIP = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "sup":
            self.out.append("^")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    p = P()
    with open(src, encoding="utf-8", errors="replace") as f:
        p.feed(f.read())
    text = "".join(p.out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
