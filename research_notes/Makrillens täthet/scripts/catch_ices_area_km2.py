"""Sea area (km2) of ICES areas from the ICES areas shapefile (EPSG:3857, land cut out).

Pure-python reader for the .shp polygons (shape type 5) and the .dbf attributes;
the file is parsed as data only. Each vertex is taken back to WGS84 lon/lat
(spherical Mercator inverse, as EPSG:3857 defines it), then to the cylindrical
equal-area projection on the WGS84 authalic sphere (x = R_q lon, y = R_q sin beta,
beta = authalic latitude), where the shoelace area of a ring is its true area on
the ellipsoid (up to edge interpolation; the 'dense' file has dense vertices).
Ring signs: shapefile outer rings are clockwise, holes counter-clockwise, so the
signed sum gives outer minus holes.

Usage: python3 -I catch_ices_area_km2.py SHP_BASENAME AREA_CODE [AREA_CODE ...]
  e.g. ... ICES_Areas_20160601_cut_dense_3857 3.a.20 3.a.21
"""
import math
import struct
import sys

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
E = math.sqrt(E2)


def q_of(phi):
    s = math.sin(phi)
    return (1 - E2) * (s / (1 - E2 * s * s) - (1 / (2 * E)) * math.log((1 - E * s) / (1 + E * s)))


QP = q_of(math.pi / 2)
RQ = A * math.sqrt(QP / 2)


def read_dbf(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        n_rec, hdr_len, rec_len = struct.unpack("<IHH", hdr[4:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            name = d[:11].split(b"\x00")[0].decode("latin-1")
            fields.append((name, d[16]))
        f.seek(hdr_len)
        recs = []
        for _ in range(n_rec):
            r = f.read(rec_len)
            pos = 1
            rec = {}
            for name, ln in fields:
                rec[name] = r[pos:pos + ln].decode("latin-1").strip()
                pos += ln
            recs.append(rec)
    return recs


def read_shp(path):
    shapes = []
    with open(path, "rb") as f:
        f.seek(100)
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            _, clen = struct.unpack(">ii", h)
            content = f.read(clen * 2)
            stype = struct.unpack("<i", content[:4])[0]
            if stype == 0:
                shapes.append([])
                continue
            nparts, npts = struct.unpack("<ii", content[36:44])
            parts = list(struct.unpack("<%di" % nparts, content[44:44 + 4 * nparts]))
            off = 44 + 4 * nparts
            pts = struct.unpack("<%dd" % (2 * npts), content[off:off + 16 * npts])
            xy = [(pts[2 * k], pts[2 * k + 1]) for k in range(npts)]
            parts.append(npts)
            shapes.append([xy[parts[k]:parts[k + 1]] for k in range(nparts)])
    return shapes


def ring_area_m2(ring):
    pts = []
    for x, y in ring:
        lon = x / A
        lat = math.pi / 2 - 2 * math.atan(math.exp(-y / A))
        beta = math.asin(q_of(lat) / QP)
        pts.append((RQ * lon, RQ * math.sin(beta)))
    s = 0.0
    for k in range(len(pts) - 1):
        s += pts[k][0] * pts[k + 1][1] - pts[k + 1][0] * pts[k][1]
    return -s / 2  # clockwise outer ring -> positive


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    base = argv[0]
    recs = read_dbf(base + ".dbf")
    shapes = read_shp(base + ".shp")
    print("fields:", list(recs[0].keys()), file=sys.stderr)
    print("area_code,sea_area_km2,n_rings")
    for code in argv[1:]:
        for rec, shp in zip(recs, shapes):
            if code in rec.values():
                a = sum(ring_area_m2(r) for r in shp) / 1e6
                print(f"{code},{a:.0f},{len(shp)}  # record {rec}")


if __name__ == "__main__":
    main(sys.argv[1:])
