"""Download one URL into a new empty directory. Usage: python3 -I fetch.py <new_dir> <url> <filename>"""
import os, sys, urllib.request
d, url, name = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(d, exist_ok=False)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-client"})
data = urllib.request.urlopen(req, timeout=90).read()
open(os.path.join(d, name), "wb").write(data)
print(len(data), data[:8])
