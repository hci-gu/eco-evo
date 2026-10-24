"""Read a remote zip by HTTP Range requests (no full download, nothing > 50 MB stored), list members or stream one
CSV member and keep rows whose text matches all filter strings. Untrusted data: parsed as text only.
Usage: python3 -I fdi_range_filter.py URL list
       python3 -I fdi_range_filter.py URL MEMBER OUT.csv FILTER1 [FILTER2 ...]"""
import io, sys, urllib.request, zipfile

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos, 8 << 20)
        if n <= 0: return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[:len(data)] = data; self.pos += len(data)
        return len(data)

url = sys.argv[1]
z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=8 << 20))
if sys.argv[2] == "list":
    for i in z.infolist():
        print(i.filename, i.file_size, i.compress_size)
    sys.exit()
member, out, filt = sys.argv[2], sys.argv[3], sys.argv[4:]
with z.open(member) as fh, open(out, "w", encoding="utf-8") as fo:
    t = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
    hdr = t.readline(); fo.write(hdr); n = k = 0
    for line in t:
        n += 1
        u = line.upper()
        if all(f.upper() in u for f in filt):
            fo.write(line); k += 1
print("rows read", n, "kept", k)
