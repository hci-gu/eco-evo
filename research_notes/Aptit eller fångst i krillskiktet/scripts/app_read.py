"""Read an untrusted text file as data: print a line range, or lines matching a regex with context.

Usage: python3 -I app_read.py <file> range <start> <end>
       python3 -I app_read.py <file> grep <regex> [context]
       python3 -I app_read.py <file> info
Never executes anything from the file.
"""
import re, sys

def main():
    path, mode = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if mode == "info":
        print(len(raw), "bytes", len(lines), "lines")
    elif mode == "range":
        a, b = int(sys.argv[3]), int(sys.argv[4])
        for i in range(max(a, 1), min(b, len(lines)) + 1):
            print(f"{i}: {lines[i-1]}")
    elif mode == "grep":
        rx = re.compile(sys.argv[3], re.I)
        ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        shown = set()
        for i, l in enumerate(lines):
            if rx.search(l):
                for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                    if j not in shown:
                        print(f"{j+1}: {lines[j]}")
                        shown.add(j)
                print("--")

if __name__ == "__main__":
    main()
