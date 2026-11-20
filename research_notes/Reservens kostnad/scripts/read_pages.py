"""Print selected pages (1-based, form-feed separated) of a pdftotext file, or the text
around a search key. Usage: read_pages.py FILE PAGE [PAGE...]  |  read_pages.py FILE --key TEXT [--before N] [--after N]"""
import re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
args = sys.argv[2:]
if args and args[0] == "--key":
    key = args[1]; before = int(args[args.index("--before")+1]) if "--before" in args else 300
    after = int(args[args.index("--after")+1]) if "--after" in args else 3000
    i = t.find(key)
    print(re.sub(r"\n\s*\n", "\n", t[max(0, i-before):i+after]) if i >= 0 else "key not found")
else:
    pages = t.split("\f")
    for p in args:
        print("=====PAGE", p); print(pages[int(p)-1])
