"""Download VOTO SeaExplorer glider light data in the box, binned server-side to 5 m per profile.

Usage: python3 -I voto_download.py   (writes ../data/voto/<datasetID>_bin5.csv.gz and ../data/voto/meta.json)

Source: Voice of the Ocean Foundation ERDDAP, https://erddap.observations.voiceoftheocean.org/erddap/
Missions: the delayed-mode datasets whose bounding box lies in 57.5-58.8 N, 9.3-11.6 E and that carry
downwelling_PAR (Biospherical MPE-PAR on 2024-2026 gliders; SeaBird OCR-504 with Ed380/490/532 on 2020-2022 gliders).
Binning: ERDDAP orderByMean("profile_index,depth/5"): arithmetic mean of each variable within each profile and
5 m depth bin; the depth column then holds the bin index k (bin = 5k to 5k+5 m; verified with mean pressure).
"""
import gzip
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "voto")
BASE = "https://erddap.observations.voiceoftheocean.org/erddap"
MISSIONS = """delayed_SEA068_M40 delayed_SEA068_M41 delayed_SEA068_M46 delayed_SEA068_M47 delayed_SHW002_M22
delayed_SEA069_M40 delayed_SEA069_M41 delayed_SEA069_M44 delayed_SEA069_M46 delayed_SEA069_M48 delayed_SEA069_M52
delayed_SEA069_M53 delayed_SEA069_M54 delayed_SEA055_M26 delayed_SEA044_M25 delayed_SEA044_M28 delayed_SEA044_M29
delayed_SEA044_M32 delayed_SEA044_M33 delayed_SEA044_M34 delayed_SEA044_M35 delayed_SEA044_M45 delayed_SEA044_M46
delayed_SEA044_M98 delayed_SEA066_M16 delayed_SEA061_M45 delayed_SEA078_M29 delayed_SEA078_M33""".split()
WANT = ["profile_index", "depth", "time", "latitude", "longitude", "pressure", "downwelling_PAR",
        "internal_temperature_PAR", "down_irradiance_380", "down_irradiance_490", "down_irradiance_532",
        "chlorophyll", "salinity", "temperature", "particulate_backscatter", "backscatter_scaled", "backscatter",
        "pitch", "profile_direction"]
KEEP_ATTR = ("irradiance", "optics", "license", "institution", "title", "comment", "summary", "creator_name",
             "publisher_name", "references", "citation", "acknowledgement", "doi")


def get(url, timeout=600):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    meta_path = os.path.join(OUT, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for ds in MISSIONS:
        out = os.path.join(OUT, f"{ds}_bin5.csv.gz")
        if os.path.exists(out) and ds in meta:
            continue
        info = get(f"{BASE}/info/{ds}/index.csv").decode()
        vars_, attrs = [], {}
        for line in info.splitlines()[1:]:
            parts = line.split(",", 4)
            if parts[0] == "variable":
                vars_.append(parts[1])
            elif parts[0] == "attribute" and parts[1] == "NC_GLOBAL" and parts[2] in KEEP_ATTR:
                attrs[parts[2]] = parts[4].strip('"')
        cols = [v for v in WANT if v in vars_]
        q = ",".join(cols) + "&downwelling_PAR!=NaN&" + urllib.parse.quote('orderByMean("profile_index,depth/5")', safe="=&")
        url = f"{BASE}/tabledap/{ds}.csv?{q}"
        data = get(url)
        gzip.open(out, "wb").write(data)
        meta[ds] = dict(url=url, columns=cols, attrs=attrs, bytes=len(data))
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(ds, len(data), "bytes", cols, flush=True)


if __name__ == "__main__":
    main()
