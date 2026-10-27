"""Display languages of the live viewer (section 212).

The viewer's strings are written in English in ``pygame_viz.py`` and stay
the source: a translation is looked up by the English template at draw
time, and a template without a translation is shown in English. Internal
state (tab ids, playback modes, status keys, spawn modes) is never
translated. Swedish and Italian are a display overlay only (AGENTS.md
section 2); the terminal output and the HTML export stay English.

Templates use ``str.format`` fields; every translation must use the same
fields as its English key (``tests/test_viz_i18n.py`` checks it).
"""

from __future__ import annotations

from typing import Optional

LANGUAGES = (("en", "English"), ("sv", "Svenska"), ("it", "Italiano"))
CODES = tuple(code for code, _name in LANGUAGES)
DEFAULT = "en"

# Full month names; the viewer abbreviates with the first three letters
# (which gives daylight.MONTH_NAMES in English).
MONTHS = {
    "en": ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"),
    "sv": ("januari", "februari", "mars", "april", "maj", "juni", "juli",
           "augusti", "september", "oktober", "november", "december"),
    "it": ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre",
           "dicembre"),
}

# Functional-group names. English comes from the FG id ("planktivorous_fish"
# -> "Planktivorous fish"), so it has no table. The Swedish names are the
# FG editor's ``sv_mapping`` (fgconfig.py imports this table).
FG_NAMES = {
    "sv": {
        "phytoplankton": "Växtplankton",
        "subsurface_phytoplankton": "Djupväxtplankton",
        "zooplankton": "Djurplankton",
        "microzooplankton": "Mikrozooplankton",
        "benthic_infauna": "Infauna",
        "mobile_epifauna": "Rörlig epifauna",
        "norway_lobster": "Havskräfta",
        "planktivorous_fish": "Planktonätande fisk",
        "benthivorous_fish": "Bottendjursätande fisk",
        "predatory_fish": "Rovfisk",
        "porpoises": "Tumlare",
        "seals": "Sälar",
        "seabirds": "Sjöfåglar",
    },
    "it": {
        "phytoplankton": "Fitoplancton",
        "subsurface_phytoplankton": "Fitoplancton profondo",
        "zooplankton": "Zooplancton",
        "microzooplankton": "Microzooplancton",
        "benthic_infauna": "Infauna bentonica",
        "mobile_epifauna": "Epifauna mobile",
        "norway_lobster": "Scampo",
        "planktivorous_fish": "Pesci planctivori",
        "benthivorous_fish": "Pesci bentivori",
        "predatory_fish": "Pesci predatori",
        "porpoises": "Focene",
        "seals": "Foche",
        "seabirds": "Uccelli marini",
    },
}

# Names in the library's ``member_species`` (keyed by the English name,
# lower case). The "others" row is formatted from its count instead.
SPECIES = {
    "sv": {
        "haddock": "kolja", "cod": "torsk", "long rough dab": "lerskädda",
        "dab": "sandskädda", "plaice": "rödspotta", "witch": "rödtunga",
        "starry ray": "klorocka", "hagfish": "pirål", "saithe": "gråsej",
        "whiting": "vitling", "mackerel": "makrill", "spurdog": "pigghaj",
        "horse mackerel": "taggmakrill", "hake": "kummel",
        "anglerfish": "marulk", "herring": "sill", "blue whiting": "kolmule",
        "norway pout": "vitlinglyra", "sprat": "skarpsill",
        "kittiwake": "tretåig mås", "guillemot": "sillgrissla",
        "razorbill": "tordmule", "gannet": "havssula",
        "gulls": "måsar och trutar", "polychaetes": "havsborstmaskar",
        "brittle stars and other echinoderms":
            "ormstjärnor och andra tagghudingar",
        "crabs and hermit crabs": "krabbor och eremitkräftor",
        "bivalves and other molluscs": "musslor och andra blötdjur",
        "northern shrimp": "nordhavsräka",
        "norway lobster": "havskräfta",
        "crabs, hermit crabs and squat lobsters":
            "krabbor, eremitkräftor och trollhumrar",
        "small shrimps": "små räkor", "mysids": "pungräkor",
        "cephalopods and small demersal fish": "bläckfiskar och små bottenfiskar",
        "amphipods and other small crustaceans":
            "märlkräftor och andra små kräftdjur",
        "ciliates": "ciliater",
        "heterotrophic dinoflagellates": "heterotrofa dinoflagellater",
        "copepods (calanus, metridia, small species)":
            "hoppkräftor (Calanus, Metridia, små arter)",
        "krill": "krill",
    },
    "it": {
        "haddock": "eglefino", "cod": "merluzzo bianco",
        "long rough dab": "passera canadese", "dab": "limanda",
        "plaice": "passera di mare", "witch": "passera lingua di cane",
        "starry ray": "razza radiata", "hagfish": "missina",
        "saithe": "merluzzo carbonaro", "whiting": "merlano",
        "mackerel": "sgombro", "spurdog": "spinarolo",
        "horse mackerel": "suro", "hake": "nasello",
        "anglerfish": "rana pescatrice", "herring": "aringa",
        "blue whiting": "melù", "norway pout": "busbana norvegese",
        "sprat": "spratto", "kittiwake": "gabbiano tridattilo",
        "guillemot": "uria", "razorbill": "gazza marina",
        "gannet": "sula", "gulls": "gabbiani", "polychaetes": "policheti",
        "brittle stars and other echinoderms": "ofiure e altri echinodermi",
        "crabs and hermit crabs": "granchi e paguri",
        "bivalves and other molluscs": "bivalvi e altri molluschi",
        "northern shrimp": "gamberetto boreale",
        "norway lobster": "scampo",
        "crabs, hermit crabs and squat lobsters":
            "granchi, paguri e galatee",
        "small shrimps": "gamberetti", "mysids": "misidacei",
        "cephalopods and small demersal fish": "cefalopodi e piccoli pesci di fondo",
        "amphipods and other small crustaceans":
            "anfipodi e altri piccoli crostacei",
        "ciliates": "ciliati",
        "heterotrophic dinoflagellates": "dinoflagellati eterotrofi",
        "copepods (calanus, metridia, small species)":
            "copepodi (Calanus, Metridia, specie piccole)",
        "krill": "krill",
    },
}

# English template -> translation. Abbreviations: move/rest/eat and
# predation/starvation/impact/natural (M1) in each language.
CATALOG = {
    "sv": {
        # Status bar.
        "mode = {mode}": "läge = {mode}",
        "train": "träning",
        "inference": "inferens",
        "time = {time}": "tid = {time}",
        "{years} y {days} d": "{years} år {days} d",
        "month = {month}": "månad = {month}",
        "daylight = {hours} h": "dagsljus = {hours} h",
        "[PAUSED]": "[PAUSAD]",
        "plot:log": "graf:log",
        "grid = {w}x{h}": "rutnät = {w}x{h}",
        "norm: rolling [n]": "norm: rullande [n]",
        "norm: fixed [n]": "norm: fast [n]",
        "Probe ticks{marker} = {value}": "Probe-ticks{marker} = {value}",
        "Perturbation ticks{marker} = {value}":
            "Perturbationsticks{marker} = {value}",
        "Evaluation probe interval{marker} = {value}":
            "Probe-intervall{marker} = {value}",
        "Biomass scale = {value}x": "Biomassaskala = {value}x",
        # Playback bar.
        "resume train": "fortsätt",
        "pause train": "pausa",
        "rec ({n})": "insp ({n})",
        "no capture": "ingen insp",
        "live  (saved frames: {n})": "live  (bilder: {n})",
        "frame = {i}/{n}": "bild = {i}/{n}",
        "recording... ({n} frames)": "spelar in... ({n} bilder)",
        "capture off": "inspelning av",
        "Decision makers": "Beslutsfattare",
        "Non-decision makers": "Icke-beslutsfattare",
        # Heatmap panels.
        "pr/im = {pr}/{im}%": "pr/pv = {pr}/{im}%",
        "mv/rs/et = {mv}/{rs}/{et}%": "fl/vi/ät = {mv}/{rs}/{et}%",
        "pr/st/im/nm = {pr}/{st}/{im}/{nm}%":
            "pr/sv/pv/nd = {pr}/{st}/{im}/{nm}%",
        "Mode: {value}": "Läge: {value}",
        "Tpl:  {value}": "Mall: {value}",
        "(default)": "(standard)",
        "Uniform": "Likformig",
        "Colony": "Koloni",
        "Env-driven": "Miljöstyrd",
        "Equilibrium": "Jämvikt",
        "others ({count} species)": "övriga ({count} arter)",
        # Plot tabs.
        "reward": "belöning",
        "biomass": "biomassa",
        "energy": "energi",
        "move": "flytt",
        "rest": "vila",
        "eat": "äta",
        "predation": "predation",
        "starvation": "svält",
        "impacts": "påverkan",
        "natural": "naturlig",
        "viability": "viabilitet",
        "Viability margin per probe (0 = bound, 1 = spawn level / no drift)":
            "Viabilitetsmarginal per prob (0 = gräns, 1 = startnivå / ingen drift)",
        "Training update": "Träningsuppdatering",
        "bound": "gräns",
        "update {n}": "uppdatering {n}",
        # Plot titles.
        "biomass (% of start)": "biomassa (% av start)",
        "energy (% of start)": "energi (% av start)",
        "avg move action (%)": "medel flytt-handling (%)",
        "avg rest action (%)": "medel vila-handling (%)",
        "avg eat action (%)": "medel ät-handling (%)",
        "move action (%)": "flytt-handling (%)",
        "rest action (%)": "vila-handling (%)",
        "eat action (%)": "ät-handling (%)",
        "predation share of total loss (%)":
            "predationens andel av total förlust (%)",
        "starvation share of total loss (%)":
            "svältens andel av total förlust (%)",
        "impact share of total loss (%)":
            "påverkans andel av total förlust (%)",
        "natural mortality (M1) share of total loss (%)":
            "naturlig dödlighet (M1), andel av total förlust (%)",
        "predation share of tick loss (%)":
            "predationens andel av tickets förlust (%)",
        "starvation share of tick loss (%)":
            "svältens andel av tickets förlust (%)",
        "impact share of tick loss (%)":
            "påverkans andel av tickets förlust (%)",
        "natural mortality (M1) share of tick loss (%)":
            "naturlig dödlighet (M1), andel av tickets förlust (%)",
        "long run": "långkörning",
        "all": "alla",
        "Long run of the current policies (biomass, % of start)":
            "Långkörning med de aktuella policyerna (biomassa, % av start)",
        "Long run of the current policies, {probe}; trained on {train} rollouts (biomass, % of start)":
            "Långkörning med de aktuella policyerna, {probe}; tränade på rollouts om {train} (biomassa, % av start)",
        "{n} year": "{n} år",
        "{n} years": "{n} år",
        "{n} day": "{n} dag",
        "{n} days": "{n} dagar",
        "Waiting for the first probe": "Väntar på den första proben",
        "No starting biomass: {names}": "Ingen startbiomassa: {names}",
        "Update {step}: frozen policies, seed {seed} | {n} probes":
            "Uppdatering {step}: frysta policyer, frö {seed} | {n} prober",
        "Viable {viable}/{present}, V {score}": "Viabla {viable}/{present}, V {score}",
        "long-run years {years}": "långkörningens år {years}",
        "worst {name}: {why}": "sämst {name}: {why}",
        "worst {name}: {margin} to {why}": "sämst {name}: {margin} till {why}",
        "floor in ~{years} yr": "golvet om ~{years} år",
        "floor": "golv",
        "ceiling": "tak",
        "drift": "drift",
        "extinct": "utdöd",
        "Progress unavailable; see terminal for details":
            "Framsteg saknas; se terminalen för detaljer",
        "ARS reward per FG": "ARS-belöning per FG",
        "{label}  display max = {pct}%": "{label}  visningsmax = {pct}%",
        "reward (log10 ratio)": "belöning (log10-kvot)",
        "Long-run year (inference start = 0)": "Långkörningens år (inferensstart = 0)",
        "long-run biomass (log10 of % of start)": "långkörningens biomassa (log10 av % av start)",
        "Save plot as HTML...": "Spara graf som HTML...",
        "No data to save": "Ingen data att spara",
        "Saved: {name}": "Sparad: {name}",
        "Save inference plot as HTML": "Spara inferensgrafen som HTML",
        "HTML files": "HTML-filer",
        "All files": "Alla filer",
        "tick {n}": "tick {n}",
        "day {n}": "dag {n}",
        "PLOT": "GRAF",
    },
    "it": {
        # Status bar.
        "mode = {mode}": "modo = {mode}",
        "train": "addestramento",
        "inference": "inferenza",
        "time = {time}": "tempo = {time}",
        "{years} y {days} d": "{years} a {days} g",
        "month = {month}": "mese = {month}",
        "daylight = {hours} h": "luce = {hours} h",
        "[PAUSED]": "[IN PAUSA]",
        "plot:log": "grafico:log",
        "grid = {w}x{h}": "griglia = {w}x{h}",
        "norm: rolling [n]": "norm: mobile [n]",
        "norm: fixed [n]": "norm: fissa [n]",
        "Probe ticks{marker} = {value}": "Tick sonda{marker} = {value}",
        "Perturbation ticks{marker} = {value}":
            "Tick perturbazione{marker} = {value}",
        "Evaluation probe interval{marker} = {value}":
            "Intervallo sonde{marker} = {value}",
        "Biomass scale = {value}x": "Scala biomassa = {value}x",
        # Playback bar.
        "resume train": "riprendi",
        "pause train": "pausa",
        "rec ({n})": "reg ({n})",
        "no capture": "no reg",
        "live  (saved frames: {n})": "live  (fotogrammi: {n})",
        "frame = {i}/{n}": "fotogr. = {i}/{n}",
        "recording... ({n} frames)": "in registrazione ({n})",
        "capture off": "registrazione spenta",
        "Decision makers": "Decisori",
        "Non-decision makers": "Non decisori",
        # Heatmap panels.
        "pr/im = {pr}/{im}%": "pr/im = {pr}/{im}%",
        "mv/rs/et = {mv}/{rs}/{et}%": "mv/rp/al = {mv}/{rs}/{et}%",
        "pr/st/im/nm = {pr}/{st}/{im}/{nm}%":
            "pr/fa/im/mn = {pr}/{st}/{im}/{nm}%",
        "Mode: {value}": "Modo: {value}",
        "Tpl:  {value}": "Schema: {value}",
        "(default)": "(predefinito)",
        "Uniform": "Uniforme",
        "Colony": "Colonia",
        "Env-driven": "Ambientale",
        "Equilibrium": "Equilibrio",
        "others ({count} species)": "altre ({count} specie)",
        # Plot tabs.
        "reward": "premio",
        "biomass": "biomassa",
        "energy": "energia",
        "move": "moto",
        "rest": "riposo",
        "eat": "pasto",
        "predation": "predazione",
        "starvation": "fame",
        "impacts": "impatti",
        "natural": "naturale",
        "viability": "vitalità",
        "Viability margin per probe (0 = bound, 1 = spawn level / no drift)":
            "Margine di vitalità per sonda (0 = soglia, 1 = livello iniziale / nessuna deriva)",
        "Training update": "Aggiornamento dell'addestramento",
        "bound": "soglia",
        "update {n}": "aggiornamento {n}",
        # Plot titles.
        "biomass (% of start)": "biomassa (% dell'inizio)",
        "energy (% of start)": "energia (% dell'inizio)",
        "avg move action (%)": "azione movimento media (%)",
        "avg rest action (%)": "azione riposo media (%)",
        "avg eat action (%)": "azione alimentazione media (%)",
        "move action (%)": "azione movimento (%)",
        "rest action (%)": "azione riposo (%)",
        "eat action (%)": "azione alimentazione (%)",
        "predation share of total loss (%)":
            "quota della predazione sulla perdita totale (%)",
        "starvation share of total loss (%)":
            "quota della fame sulla perdita totale (%)",
        "impact share of total loss (%)":
            "quota degli impatti sulla perdita totale (%)",
        "natural mortality (M1) share of total loss (%)":
            "quota della mortalità naturale (M1) sulla perdita totale (%)",
        "predation share of tick loss (%)":
            "quota della predazione sulla perdita del tick (%)",
        "starvation share of tick loss (%)":
            "quota della fame sulla perdita del tick (%)",
        "impact share of tick loss (%)":
            "quota degli impatti sulla perdita del tick (%)",
        "natural mortality (M1) share of tick loss (%)":
            "quota della mortalità naturale (M1) sulla perdita del tick (%)",
        "long run": "lungo periodo",
        "all": "tutti",
        "Long run of the current policies (biomass, % of start)":
            "Lungo periodo delle politiche attuali (biomassa, % dell'inizio)",
        "Long run of the current policies, {probe}; trained on {train} rollouts (biomass, % of start)":
            "Lungo periodo delle politiche attuali, {probe}; addestrate su rollout di {train} (biomassa, % dell'inizio)",
        "{n} year": "{n} anno",
        "{n} years": "{n} anni",
        "{n} day": "{n} giorno",
        "{n} days": "{n} giorni",
        "Waiting for the first probe": "In attesa della prima sonda",
        "No starting biomass: {names}": "Nessuna biomassa iniziale: {names}",
        "Update {step}: frozen policies, seed {seed} | {n} probes":
            "Aggiornamento {step}: politiche congelate, seme {seed} | {n} sonde",
        "Viable {viable}/{present}, V {score}": "Vitali {viable}/{present}, V {score}",
        "long-run years {years}": "anni del lungo periodo {years}",
        "worst {name}: {why}": "peggiore {name}: {why}",
        "worst {name}: {margin} to {why}": "peggiore {name}: {margin} da {why}",
        "floor in ~{years} yr": "soglia minima tra ~{years} anni",
        "floor": "soglia minima",
        "ceiling": "soglia massima",
        "drift": "deriva",
        "extinct": "estinto",
        "Progress unavailable; see terminal for details":
            "Progressi non disponibili; vedi il terminale",
        "ARS reward per FG": "Premio ARS per FG",
        "{label}  display max = {pct}%": "{label}  massimo mostrato = {pct}%",
        "reward (log10 ratio)": "premio (rapporto log10)",
        "Long-run year (inference start = 0)": "Anno del lungo periodo (inizio dell'inferenza = 0)",
        "long-run biomass (log10 of % of start)": "biomassa del lungo periodo (log10 della % dell'inizio)",
        "Save plot as HTML...": "Salva grafico HTML...",
        "No data to save": "Nessun dato da salvare",
        "Saved: {name}": "Salvato: {name}",
        "Save inference plot as HTML": "Salva il grafico di inferenza come HTML",
        "HTML files": "File HTML",
        "All files": "Tutti i file",
        "tick {n}": "tick {n}",
        "day {n}": "giorno {n}",
        "PLOT": "GRAFICO",
    },
}


def translate(lang: str, text: str, **fields) -> str:
    """``text`` (an English template) in ``lang``, formatted with ``fields``."""
    out = CATALOG.get(lang, {}).get(text, text)
    return out.format(**fields) if fields else out


def fg_name(lang: str, fid: str) -> Optional[str]:
    """The FG's name in ``lang``, or None where the id itself is the name."""
    return FG_NAMES.get(lang, {}).get(fid)


def species_name(lang: str, name: str) -> str:
    """A ``member_species`` name in ``lang`` (unknown names unchanged)."""
    return SPECIES.get(lang, {}).get(str(name).strip().lower(), name)


def month_name(lang: str, month: int) -> str:
    """Full month name (``month`` 0-based), capitalised for a label."""
    name = MONTHS.get(lang, MONTHS["en"])[int(month) % 12]
    return name[:1].upper() + name[1:]


# ---- Flags ---------------------------------------------------------------
_UK_BLUE, _UK_RED, _WHITE = (1, 33, 105), (200, 16, 46), (255, 255, 255)
_SE_BLUE, _SE_YELLOW = (0, 106, 167), (254, 204, 0)
_IT_GREEN, _IT_WHITE, _IT_RED = (0, 146, 70), (244, 245, 240), (206, 43, 55)


def draw_flag(pg, surface, rect, lang: str) -> None:
    """Draw the flag of ``lang`` into ``rect`` (x, y, w, h), with a frame.

    Drawn with primitives, so the viewer needs no image files: the Union
    Jack (English), Sweden and Italy.
    """
    x, y, w, h = (int(v) for v in rect)
    old_clip = surface.get_clip()
    surface.set_clip(pg.Rect(x, y, w, h).clip(old_clip) if old_clip else
                     pg.Rect(x, y, w, h))
    try:
        if lang == "sv":
            surface.fill(_SE_BLUE, (x, y, w, h))
            bar = max(2, round(h * 0.2))
            vx = x + round(w * 5 / 16)
            surface.fill(_SE_YELLOW, (vx, y, bar, h))
            surface.fill(_SE_YELLOW, (x, y + (h - bar) // 2, w, bar))
        elif lang == "it":
            third = w / 3.0
            surface.fill(_IT_GREEN, (x, y, round(third), h))
            surface.fill(_IT_WHITE, (x + round(third), y,
                                     round(2 * third) - round(third), h))
            surface.fill(_IT_RED, (x + round(2 * third), y,
                                   w - round(2 * third), h))
        else:
            surface.fill(_UK_BLUE, (x, y, w, h))
            x1, y1 = x + w - 1, y + h - 1
            diag = max(2, round(h / 5))
            for a, b in (((x, y), (x1, y1)), ((x, y1), (x1, y))):
                pg.draw.line(surface, _WHITE, a, b, diag + 1)
                pg.draw.line(surface, _UK_RED, a, b, max(1, diag // 2))
            cross = max(3, round(h / 3))
            red = max(1, round(h / 5))
            surface.fill(_WHITE, (x + (w - cross) // 2, y, cross, h))
            surface.fill(_WHITE, (x, y + (h - cross) // 2, w, cross))
            surface.fill(_UK_RED, (x + (w - red) // 2, y, red, h))
            surface.fill(_UK_RED, (x, y + (h - red) // 2, w, red))
    finally:
        surface.set_clip(old_clip)
    pg.draw.rect(surface, (150, 150, 165), (x - 1, y - 1, w + 2, h + 2), 1)
