"""Acoustic (HERAS, RV Dana) biomass density per species in the Mareld box.

Usage:
  python3 -I estimate_box.py NASC_LOGS HAULS CATCH INDIVIDUALS OUTDIR [--boot N]

Inputs are the tables written by acoustic_box.py and biotic_region.py (derived from
untrusted ICES downloads; parsed as data only).

Method (a simplified StoX-style echo-integration estimate) [D]:
  1. Box mean NASC per SaCategory = distance-weighted mean over valid logs inside the box.
  2. Trawl hauls inside the allocation region (box + ~25 km) give, per haul, the
     catch weight per species; numbers = weight / mean individual weight; sigma_bs
     from TS = 20 log10(L cm) + b20 averaged over the length distribution.
  3. Each haul's backscatter shares s_ih = n_i sigma_i / sum_j n_j sigma_j; the NASC
     share of species i is the haul-equal mean of s_ih.  Category MIX is split over all
     acoustic target species, CLU over clupeids only, HER/SPR/PIL/ANE are taken as is.
  4. Density (n/nmi2) = NASC_i / (4 pi <sigma_i>); biomass = density * mean weight;
     t/km2 = kg/nmi2 / 1000 / 3.4299.
  5. Years without biotic data (2010-2015) borrow the pooled 2016-2020 hauls.
  6. Bootstrap: resample transect segments (logs split at gaps > 2 h) and hauls.
"""
import csv, math, os, random, sys, collections

BOX = (57.7, 58.4, 9.9, 11.3)          # lat0, lat1, lon0, lon1 (Mareld grid)
HREG = (57.3, 58.8, 9.3, 11.9)         # haul allocation region
NMI2_KM2 = 1.852 ** 2
INCLUDE_MACKEREL = '--mackerel' in sys.argv      # sensitivity: let mackerel take MIX backscatter
FOT_ONLY = '--fot-only' in sys.argv                # sensitivity: pelagic (FOT) hauls only
if '--hauls-in-box' in sys.argv:                   # sensitivity: hauls inside the box only
    HREG = BOX

# TS b20 (dB re 1 m2, L in cm) by AphiaID; HERAS / WGIPS standard values
CLUPEIDS = {'126417': 'herring', '126425': 'sprat', '126421': 'sardine', '126426': 'anchovy',
            '126413': 'allis_shad', '126415': 'twaite_shad'}
GADOID_LIKE = {'126444': 'norway_pout', '126438': 'whiting', '126437': 'haddock', '126436': 'cod',
               '126441': 'saithe', '126440': 'pollack', '126446': 'poor_cod', '126439': 'blue_whiting',
               '126435': 'silvery_pout', '126484': 'hake', '126822': 'horse_mackerel',
               '126715': 'greater_argentine', '126716': 'lesser_argentine', '127312': 'pearlside',
               '126375': 'garfish', '126461': 'ling', '126450': 'fourbeard_rockling'}
MACKEREL = {'127023': 'mackerel'}
B20 = {}
B20.update({k: -71.2 for k in CLUPEIDS})
B20.update({k: -67.5 for k in GADOID_LIKE})
B20.update({k: -84.9 for k in MACKEREL})
NAME = {}
NAME.update(CLUPEIDS); NAME.update(GADOID_LIKE); NAME.update(MACKEREL)
LMAX = {'126417': 400, '126425': 200, '127023': 500, '126444': 260, '126439': 500, '127312': 80}
CAT_SPECIES = {'HER': ['126417'], 'SPR': ['126425'], 'PIL': ['126421'], 'ANE': ['126426']}
PLANKTIVORES = ['herring', 'sprat', 'norway_pout', 'sardine', 'anchovy', 'pearlside', 'lesser_argentine']


GRID = '--grid' in sys.argv   # the true 60 x 60 km grid (SWEREF 99 TM E 211-271 km, N 6414-6474 km)
if GRID:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools'))
    from make_fishing_maps import wgs84_to_sweref99tm  # repo code, not downloaded


def inside(lat, lon, b):
    if GRID and b is BOX:
        e, n = wgs84_to_sweref99tm(lon, lat)
        return 211000 <= e < 271000 and 6414000 <= n < 6474000
    return b[0] <= lat <= b[1] and b[2] <= lon <= b[3]


def year_of(cruise, time):
    return int(time[:4])


def load_logs(path, box):
    logs = collections.defaultdict(list)  # year -> list of (time, interval, {cat: nasc}, lat, lon)
    tmp = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['survey'] != 'HERAS' or r['country'] != 'DK':
            continue
        if r['validity'] not in ('V', ''):
            continue
        lat, lon = float(r['lat']), float(r['lon'])
        if not inside(lat, lon, box):
            continue
        key = (r['cruise'], r['log_time'], r['lat'], r['lon'])
        tmp[key][r['category']] = tmp[key].get(r['category'], 0.0) + float(r['nasc'])
        try:
            iv = float(r['interval_nmi'])
        except ValueError:
            iv = 1.0
        meta[key] = (iv, lat, lon)
    for key, cats in tmp.items():
        iv, lat, lon = meta[key]
        logs[int(key[1][:4])].append((key[1], iv, cats, lat, lon))
    for y in logs:
        logs[y].sort(key=lambda r: (r[0], r[3], r[4]))
    return logs


def segments(loglist, gap_h=2.0):
    from datetime import datetime
    segs, cur, last = [], [], None
    for rec in loglist:
        t = datetime.fromisoformat(rec[0][:16])
        if last is not None and (t - last).total_seconds() > gap_h * 3600:
            segs.append(cur); cur = []
        cur.append(rec); last = t
    if cur:
        segs.append(cur)
    return segs


def mean_nasc(segs):
    num = collections.Counter(); den = 0.0
    for s in segs:
        for (_, iv, cats, _, _) in s:
            den += iv
            for c, v in cats.items():
                num[c] += v * iv
    return {c: num[c] / den for c in num} if den > 0 else {}, den


def load_hauls(hpath, cpath, ipath):
    hauls = {}
    for r in csv.DictReader(open(hpath), delimiter='\t'):
        if not (r['cruise'].startswith('26D') or r['cruise'].startswith('Dana')):
            continue
        if r['validity'] not in ('V', ''):
            continue
        if FOT_ONLY and r['gear'] != 'FOT':
            continue
        if inside(float(r['lat']), float(r['lon']), HREG):
            hauls[(r['cruise'], r['haul'])] = {'year': int(r['time'][:4]), 'gear': r['gear'], 'W': collections.Counter()}
    for r in csv.DictReader(open(cpath), delimiter='\t'):
        k = (r['cruise'], r['haul'])
        if k in hauls and r['species'] in B20:
            hauls[k]['W'][r['species']] += float(r['weight_kg'] or 0)
    ind = collections.defaultdict(list)   # (cruise, haul, sp) -> [(L mm, w kg or None)]
    for r in csv.DictReader(open(ipath), delimiter='\t'):
        sp = r['species']
        if sp not in B20:
            continue
        L = float(r['length_mm'])
        if L <= 0 or L > LMAX.get(sp, 1500):
            continue
        w = float(r['weight_kg']) if r['weight_kg'] else None
        if w is not None and (w <= 0 or w > 30):
            w = None
        ind[(r['cruise'], r['haul'], sp)].append((L, w))
    return hauls, ind


def _fit(pts):
    n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return my - b * mx, b


def lw_fit(pairs):
    """log-log length-weight fit (g, cm), refitted after dropping |residual| > 0.5."""
    pts = [(math.log(L / 10), math.log(w * 1000)) for L, w in pairs if w]
    if len(pts) < 10:
        return None
    f = _fit(pts)
    if f is None:
        return None
    for _ in range(2):
        keep = [p for p in pts if abs(p[1] - f[0] - f[1] * p[0]) < 0.5]
        if len(keep) < 10:
            return None
        f = _fit(keep)
    if not (2.5 <= f[1] <= 3.6):
        return None
    return f


def species_stats(ind_list, b20, fit):
    """mean weight (kg) and mean sigma_bs (m2) of a list of (L mm, w kg)."""
    if not ind_list:
        return None
    sig = 0.0; wsum = 0.0
    for L, w in ind_list:
        Lc = L / 10.0
        sig += 10 ** ((20 * math.log10(Lc) + b20) / 10)
        wfit = (math.exp(fit[0]) * Lc ** fit[1] if fit else 0.01 * Lc ** 3) / 1000.0
        if w is None or not (0.5 < w / wfit < 2.0):
            w = wfit
        wsum += w
    n = len(ind_list)
    return wsum / n, sig / n


def haul_tables(hauls, ind):
    """Per haul: {sp: (n, sigma, wbar)} plus pooled per-year species stats."""
    by_year_sp = collections.defaultdict(list)
    for (cr, h, sp), lst in ind.items():
        if (cr, h) in hauls:
            by_year_sp[(hauls[(cr, h)]['year'], sp)].extend(lst)
    fits = {}
    allsp = collections.defaultdict(list)
    for (y, sp), lst in by_year_sp.items():
        allsp[sp].extend(lst)
    for sp, lst in allsp.items():
        fits[sp] = lw_fit(lst)
    pooled = {}
    for (y, sp), lst in by_year_sp.items():
        pooled[(y, sp)] = species_stats(lst, B20[sp], fits.get(sp))
    out = {}
    for k, h in hauls.items():
        t = {}
        for sp, W in h['W'].items():
            if W <= 0:
                continue
            st = species_stats(ind.get(k + (sp,), []), B20[sp], fits.get(sp)) if len(ind.get(k + (sp,), [])) >= 5 else None
            if st is None:
                st = pooled.get((h['year'], sp))
            if st is None:
                # no length data this year: typical length per species
                Ltyp = {'127312': 50, '126375': 600, '126444': 120, '126438': 250, '126437': 300, '126436': 400, '126441': 500,
                        '126439': 250, '126822': 250, '127023': 280, '126417': 190, '126425': 110}.get(sp, 200)
                st = species_stats([(Ltyp, None)], B20[sp], fits.get(sp))
            wbar, sig = st
            t[sp] = (W / wbar, sig, wbar)
        if t:
            out[k] = t
    return out


def allocate(nasc, haul_list, tables):
    """Return {species: t/km2} for one year given box NASC by category and the hauls."""
    res = collections.Counter()
    for cat, val in nasc.items():
        if val <= 0:
            continue
        if cat in CAT_SPECIES:
            pool = CAT_SPECIES[cat]
        elif cat == 'CLU':
            pool = list(CLUPEIDS)
        else:  # MIX and anything else; mackerel (no swimbladder) is left out of MIX
            pool = [k for k in B20 if k not in MACKEREL or INCLUDE_MACKEREL]
        shares = collections.Counter(); nh = 0
        num = collections.Counter(); sig_n = collections.Counter(); w_n = collections.Counter()
        for k in haul_list:
            t = {sp: v for sp, v in tables[k].items() if sp in pool}
            tot = sum(n * s for n, s, _ in t.values())
            if tot <= 0:
                continue
            nh += 1
            for sp, (n, s, w) in t.items():
                shares[sp] += n * s / tot
                num[sp] += n; sig_n[sp] += n * s; w_n[sp] += n * w
        if nh == 0:
            if len(pool) == 1:
                continue
            continue
        for sp in shares:
            p = shares[sp] / nh
            sig = sig_n[sp] / num[sp]; wbar = w_n[sp] / num[sp]
            dens = val * p / (4 * math.pi * sig)          # n / nmi2
            res[NAME[sp]] += dens * wbar / 1000.0 / NMI2_KM2  # t / km2
    return res


def main():
    nasc_p, hp, cp, ip, outdir = sys.argv[1:6]
    nboot = int(sys.argv[sys.argv.index('--boot') + 1]) if '--boot' in sys.argv else 500
    tag = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
    rng = random.Random(1)
    logs = load_logs(nasc_p, BOX)
    hauls, ind = load_hauls(hp, cp, ip)
    tables = haul_tables(hauls, ind)
    hy = collections.defaultdict(list)
    for k in tables:
        hy[hauls[k]['year']].append(k)
    borrow = [k for y in range(2016, 2021) for k in hy.get(y, [])]
    for y in range(2010, 2016):
        if not hy.get(y):
            hy[y] = borrow
    borrowed = {y for y in range(2010, 2016)}
    for y in logs:
        if not hy.get(y):   # e.g. 2026: biotic not yet submitted -> previous year
            prev = max(k for k in hy if hy[k] and k < y)
            hy[y] = hy[prev]; borrowed.add(y)
    species_order = ['herring', 'sprat', 'norway_pout', 'mackerel', 'blue_whiting', 'whiting', 'haddock',
                     'saithe', 'cod', 'horse_mackerel', 'sardine', 'anchovy', 'pearlside', 'hake']
    os.makedirs(outdir, exist_ok=True)
    fy = open(os.path.join(outdir, f'acoustic_box_by_year{tag}.csv'), 'w', newline='')
    wy = csv.writer(fy)
    wy.writerow(['year', 'cruise_logs_in_box', 'nmi_in_box', 'segments', 'hauls_used', 'hauls_borrowed',
                 'nasc_total', 'nasc_by_category', 'herring_t_km2', 'herring_p05', 'herring_p95',
                 'sprat_t_km2', 'sprat_p05', 'sprat_p95', 'planktivores_t_km2', 'plank_p05', 'plank_p95',
                 'norway_pout_t_km2', 'mackerel_t_km2', 'blue_whiting_t_km2', 'other_fish_t_km2', 'all_fish_t_km2'])
    fs = open(os.path.join(outdir, f'acoustic_box_species_by_year{tag}.csv'), 'w', newline='')
    ws = csv.writer(fs)
    ws.writerow(['year', 'species', 't_km2'])
    for y in sorted(logs):
        segs = segments(logs[y])
        nasc, nmi = mean_nasc(segs)
        hl = hy.get(y, [])
        res = allocate(nasc, hl, tables)
        for sp, v in sorted(res.items(), key=lambda x: -x[1]):
            ws.writerow([y, sp, f'{v:.4f}'])
        boots = []
        for _ in range(nboot):
            bs = [segs[rng.randrange(len(segs))] for _ in segs]
            bh = [hl[rng.randrange(len(hl))] for _ in hl] if hl else []
            bn, _ = mean_nasc(bs)
            boots.append(allocate(bn, bh, tables))

        def q(name, p):
            vals = sorted(b.get(name, 0.0) if name != 'P' else sum(b.get(s, 0.0) for s in PLANKTIVORES) for b in boots)
            return vals[int(p * (len(vals) - 1))]
        plank = sum(res.get(s, 0.0) for s in PLANKTIVORES)
        allf = sum(res.values())
        other = allf - plank - res.get('mackerel', 0) - res.get('blue_whiting', 0)
        wy.writerow([y, sum(len(s) for s in segs), f'{nmi:.0f}', len(segs), len(hl), 'yes' if y in borrowed else 'no',
                     f'{sum(nasc.values()):.0f}', ';'.join(f'{c}={v:.0f}' for c, v in sorted(nasc.items())),
                     f"{res.get('herring', 0):.2f}", f"{q('herring', .05):.2f}", f"{q('herring', .95):.2f}",
                     f"{res.get('sprat', 0):.2f}", f"{q('sprat', .05):.2f}", f"{q('sprat', .95):.2f}",
                     f'{plank:.2f}', f"{q('P', .05):.2f}", f"{q('P', .95):.2f}",
                     f"{res.get('norway_pout', 0):.3f}", f"{res.get('mackerel', 0):.3f}", f"{res.get('blue_whiting', 0):.3f}",
                     f'{other:.2f}', f'{allf:.2f}'])
    fy.close(); fs.close()
    # haul composition summary
    fh = open(os.path.join(outdir, f'allocation_hauls_summary{tag}.csv'), 'w', newline='')
    wh = csv.writer(fh)
    wh.writerow(['year', 'n_hauls', 'gears', 'species', 'catch_kg', 'mean_weight_g', 'mean_TS_dB', 'mean_backscatter_share'])
    for y in sorted(set(hauls[k]['year'] for k in tables)):
        hl = [k for k in tables if hauls[k]['year'] == y]
        gears = collections.Counter(hauls[k]['gear'] for k in hl)
        agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
        for k in hl:
            tot = sum(n * s for n, s, _ in tables[k].values())
            for sp, (n, s, w) in tables[k].items():
                a = agg[sp]; a[0] += n * w; a[1] += n; a[2] += n * s; a[3] += n * w; a[4] += n * s / tot if tot else 0
        for sp, a in sorted(agg.items(), key=lambda x: -x[1][4]):
            wh.writerow([y, len(hl), dict(gears), NAME[sp], f'{a[0]:.1f}', f'{1000 * a[3] / a[1]:.1f}',
                         f'{10 * math.log10(a[2] / a[1]):.1f}', f'{a[4] / len(hl):.3f}'])
    fh.close()


if __name__ == '__main__':
    main()
