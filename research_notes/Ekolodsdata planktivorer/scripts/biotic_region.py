"""Extract hauls and catches from ICES biotic CSV zips (untrusted data).

Usage: python3 -I biotic_region.py OUT_PREFIX zip1 zip2 ...
Writes OUT_PREFIX_hauls.tsv and OUT_PREFIX_catch.tsv for hauls in 54.5-59.5 N, 5-13 E,
and OUT_PREFIX_lengths.tsv (length class in mm, count) from Catch length records or,
when those are missing, from Biology individuals."""
import csv, io, sys, zipfile, collections

def records(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        text = z.read(name).decode('utf-8-sig', errors='replace')
    hdr = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        if row[1] == 'Header':
            hdr[row[0]] = row[2:]
        elif row[1] == 'Record' and row[0] in hdr:
            yield row[0], dict(zip(hdr[row[0]], row[2:]))

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def len_mm(code, cls):
    v = f(cls)
    if v is None:
        return None
    return v if code in ('mm', 'halfcm', '') else v * 10 if code == 'cm' else v

pre = sys.argv[1]
H = open(pre + '_hauls.tsv', 'w'); C = open(pre + '_catch.tsv', 'w'); L = open(pre + '_lengths.tsv', 'w')
H.write('cruise\thaul\tgear\ttime\tlat\tlon\trect\tmin_depth\tmax_depth\tbottom\tvalidity\tstratum\n')
C.write('cruise\thaul\tspecies\tcategory\tweight_kg\tnumber\tlen_records\n')
L.write('cruise\thaul\tspecies\tlength_mm\tcount\tsource\n')
ind = open(pre + '_individuals.tsv', 'w')
ind.write('cruise\thaul\tspecies\tlength_mm\tweight_kg\n')
for zp in sys.argv[2:]:
    hauls = {}; catch = collections.defaultdict(lambda: [0.0, 0.0, 0]); clen = collections.defaultdict(float); blen = collections.defaultdict(float)
    seen_cat = set()
    for kind, r in records(zp):
        if kind == 'Haul':
            lat, lon = f(r.get('HaulStartLatitude')), f(r.get('HaulStartLongitude'))
            if lat is None or not (54.5 <= lat <= 59.5 and 5 <= lon <= 13):
                continue
            key = (r['CruiseLocalID'], r['HaulNumber'])
            hauls[key] = r
        elif kind == 'Catch':
            key = (r['CruiseLocalID'], r['HaulNumber'])
            sp = r['CatchSpeciesCode']; cat = r.get('CatchSpeciesCategory')
            k = key + (sp, cat)
            w = f(r.get('CatchSpeciesCategoryWeight')); unit = r.get('CatchWeightUnit')
            if w is not None and unit == 'g':
                w /= 1000.0
            sub_n = f(r.get('CatchSubsampledNumber')); sub_w = f(r.get('CatchSubsampleWeight')); fac = f(r.get('CatchSubsamplingFactor'))
            nal = f(r.get('CatchNumberAtLength'))
            if k not in seen_cat:
                seen_cat.add(k)
                catch[k][0] = w or 0.0
                if fac:
                    raise_ = fac
                elif sub_w and w:
                    sw = sub_w / 1000.0 if unit == 'g' else sub_w
                    raise_ = w / sw if sw > 0 else 1.0
                else:
                    raise_ = 1.0
                catch[k].append(raise_)
                if nal is None and sub_n is not None:
                    catch[k][1] = sub_n * raise_
            if nal is not None:
                lm = len_mm(r.get('CatchLengthCode'), r.get('CatchLengthClass'))
                raise_ = catch[k][3]
                catch[k][1] += nal * raise_; catch[k][2] += 1
                if lm is not None:
                    clen[key + (sp, lm)] += nal * raise_
        elif kind == 'Biology':
            key = (r['CruiseLocalID'], r['HaulNumber'])
            lm = len_mm(r.get('BiologyLengthCode'), r.get('BiologyLengthClass'))
            wt = f(r.get('BiologyIndividualWeight'))
            if wt is not None and r.get('BiologyWeightUnit') in ('g', 'gr'):
                wt /= 1000.0
            if lm is not None and key in hauls:
                blen[key + (r['CatchSpeciesCode'], lm)] += 1
                ind.write('\t'.join(map(str, [key[0], key[1], r['CatchSpeciesCode'], lm, '' if wt is None else wt])) + '\n')
    for key, r in hauls.items():
        H.write('\t'.join([key[0], key[1], r.get('HaulGear', ''), r.get('HaulStartTime', ''), r.get('HaulStartLatitude', ''), r.get('HaulStartLongitude', ''), r.get('HaulStatisticalRectangle', ''), r.get('HaulMinTrawlDepth', ''), r.get('HaulMaxTrawlDepth', ''), r.get('HaulBottomDepth', ''), r.get('HaulValidity', ''), r.get('HaulStratum', '')]) + '\n')
    for k, v in catch.items():
        if k[:2] in hauls:
            C.write('\t'.join(map(str, [k[0], k[1], k[2], k[3], v[0], v[1], v[2]])) + '\n')
    havec = set(k[:3] for k in clen)
    for k, n in clen.items():
        if k[:2] in hauls:
            L.write('\t'.join(map(str, list(k) + [n, 'catch'])) + '\n')
    for k, n in blen.items():
        if k[:2] in hauls and k[:3] not in havec:
            L.write('\t'.join(map(str, list(k) + [n, 'biology'])) + '\n')
