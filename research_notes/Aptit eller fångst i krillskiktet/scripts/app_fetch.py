"""Download one URL into a new directory as raw bytes (generic client name, no contact details).

Usage: python3 -I app_fetch.py <new_dir> <url> <filename>
Prints size and the first bytes; never executes or parses the content.
"""
import os, sys, urllib.request

def main():
    out, url, name = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
    try:
        data = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print("ERR", e); return
    p = os.path.join(out, name)
    open(p, "wb").write(data)
    print(p, len(data), data[:8])

if __name__ == "__main__":
    main()
