"""Hover help for the FG editor's input fields.

``FIELD_HELP`` maps a field key to a text that explains what the value
does in the simulated ecosystem: the mechanism (formula where there is
one), its units and what 0 / empty mean, the ecological reading, and
what else it switches or invalidates. ``Tooltip`` shows such a text in
a small borderless window while the pointer rests on a widget.

Keys are the library/manifest parameter names where one exists
(``max_energy_reserve``, matrix ``data_key``s, spawn parameter keys);
fields that have no single parameter name get a descriptive key
(``ref_grid``, ``impact_observable``, ...). A key that differs in
meaning between decision makers and non decision makers has an
``ndm:`` variant.
"""

import tkinter as tk


class Tooltip:
    """A delayed hover tooltip bound to one widget.

    Only one tooltip is visible at a time (``_active``). The tip hides on
    leave, on any click and - via ``hide_active`` - when the surrounding
    canvas scrolls, so it never floats over content that moved.
    """

    DELAY_MS = 500
    WRAP_PX = 460
    _active = None

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text  # str, or a zero-argument callable returning str
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")
        widget.bind("<Destroy>", self._cancel, add="+")

    @classmethod
    def hide_active(cls, *_args):
        if cls._active is not None:
            cls._active._cancel()

    def _resolve_text(self):
        text = self.text() if callable(self.text) else self.text
        return (text or "").strip()

    def _schedule(self, _event=None):
        self._cancel()
        try:
            self._after_id = self.widget.after(self.DELAY_MS, self._show)
        except tk.TclError:
            self._after_id = None

    def _cancel(self, _event=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None
        if Tooltip._active is self:
            Tooltip._active = None

    def _show(self):
        self._after_id = None
        text = self._resolve_text()
        if not text:
            return
        try:
            if not self.widget.winfo_exists():
                return
        except tk.TclError:
            return
        Tooltip.hide_active()
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tk.Label(tip, text=text, justify="left", wraplength=self.WRAP_PX,
                 background="#ffffe8", foreground="#202020",
                 relief="solid", borderwidth=1, padx=8, pady=6).pack()
        tip.update_idletasks()
        # Below-right of the pointer, kept on screen.
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 18
        w, h = tip.winfo_reqwidth(), tip.winfo_reqheight()
        sw, sh = tip.winfo_screenwidth(), tip.winfo_screenheight()
        if x + w > sw - 4:
            x = max(4, sw - w - 4)
        if y + h > sh - 4:
            y = max(4, self.widget.winfo_pointery() - h - 12)
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip
        Tooltip._active = self


def help_text(key):
    """The help text for ``key`` (empty if the key has none)."""
    return FIELD_HELP.get(key, "")


def attach_help(widgets, key=None, *, text=None, prefix=None):
    """Bind the help for ``key`` (or an explicit ``text``) to ``widgets``.

    ``widgets`` is one widget or an iterable of them. ``prefix`` may be a
    string or a zero-argument callable whose result is put above the
    help text (e.g. the predator -> prey pair of a matrix cell). Labels
    get the question-arrow cursor so the user can see help is there.
    """
    body = text if text is not None else help_text(key)
    if not body:
        return
    if prefix is None:
        content = body
    else:
        def content(prefix=prefix, body=body):
            head = prefix() if callable(prefix) else prefix
            return f"{head}\n\n{body}" if head else body
    if isinstance(widgets, (list, tuple)):
        items = widgets
    else:
        items = [widgets]
    for w in items:
        if w is None:
            continue
        Tooltip(w, content)
        if w.winfo_class() in ("TLabel", "Label"):
            try:
                w.configure(cursor="question_arrow")
            except tk.TclError:
                pass


# Abbreviations used below: B = biomass (t/cell), R = energy reserve
# (MJ/cell), ME = Max Energy Reserve, s = R/(B*ME) the reserve fill,
# u = Maintenance Level, DM / NDM = (non) decision maker. "Per tick"
# values are calibrated for the library tick of 6 h; a run with another
# --tick-length converts them in memory.
FIELD_HELP = {
    # ------------------------------------------------------------------
    # Project info
    # ------------------------------------------------------------------
    "project_name": (
        "Name of the project, stored as project_metadata.name in the "
        "project file. Used for display only; it has no effect on the "
        "simulation. The --run-name of train.py, not this name, decides "
        "where results are written."
    ),
    "ref_grid": (
        "Reference grid (cells, X x Y; 1 cell = 1 km2).\n\n"
        "Every biomass total entered in this editor (Initial Total "
        "Biomass Range and the Inference tab's Initial Biomass) and the "
        "Colony spawn's N colonies are meant for a grid of exactly this "
        "size. When a run uses a different grid they are scaled by "
        "(actual cells / reference cells), with a floor of 1 t on the "
        "scaled lower bound, so the biomass DENSITY stays the same.\n\n"
        "Not scaled: carrying capacity (already per cell), Perlin scale "
        "and colony sigma (in cells). Minimum 3 x 3; default 60 x 60."
    ),
    "daylight_enabled": (
        "Daylight calendar (simulation_settings.daylight).\n\n"
        "When on, every tick has a date and a sun elevation computed "
        "from the latitude, and:\n"
        "- predator-prey pairs with a Dark Ratio < 1 hunt with an "
        "attack rate that follows the light (annual mean = the library "
        "value);\n"
        "- producers with a Light Saturation grow with the light in the "
        "mixed layer (needs the Light climate below);\n"
        "- the visual part of exposure-weighted M1 follows the light;\n"
        "- Water temperature becomes possible.\n\n"
        "It also adds one input (the light level) to every decision "
        "maker's policy, so checkpoints trained with the other setting "
        "no longer load: retraining is required."
    ),
    "daylight_latitude": (
        "Latitude of the site in degrees (-90..90, north positive). "
        "Required when the Daylight calendar is on. Sets day length and "
        "sun elevation through the year, and with them the light-"
        "dependent attack rates and producer growth. High latitude = "
        "long summer days and long winter nights."
    ),
    "daylight_start_day": (
        "Day of year (1-365) at which every rollout starts, or "
        "'random'.\n\n"
        "A number starts at that day's midnight. 'random' draws the "
        "start uniformly over the year from the spawn seed, so all "
        "rollouts of one generation share the season while successive "
        "generations see different seasons - the policies then learn "
        "behaviour for the whole year rather than one season."
    ),
    "light_climate": (
        "The site's underwater light climate, used only by light-limited "
        "producers (an NDM with Light Saturation > 0) when the Daylight "
        "calendar is on. Kd(PAR), the monthly mixed-layer depths and the "
        "monthly cloud transmission must be given together (all three "
        "or none)."
    ),
    "daylight_kd": (
        "Diffuse attenuation coefficient of photosynthetically active "
        "light, Kd(PAR), in 1/m (> 0).\n\n"
        "Light at depth z is I(z) = I0 * exp(-Kd * z); mareld2.yaml "
        "uses 0.29. A higher Kd darkens the "
        "mixed layer and lowers producer growth. Part of the Light "
        "climate (all three fields or none)."
    ),
    "daylight_mixed_layer": (
        "Mixed-layer depth in metres for each month, 12 comma-separated "
        "values Jan..Dec (> 0; mid-month values, interpolated "
        "circularly).\n\n"
        "Phytoplankton are mixed through this layer, so their light-"
        "limited growth is the photosynthesis averaged over 0..z_mix. A "
        "deep winter mixed layer keeps growth low even on bright days; "
        "spring shoaling is what lets the bloom start."
    ),
    "daylight_cloud": (
        "Fraction of clear-sky light that reaches the sea surface, for "
        "each month: 12 comma-separated values Jan..Dec in (0, 1]. "
        "Multiplies the clear-sky PAR from the sun's elevation. Lower "
        "values = cloudier months = less light for producer growth."
    ),
    "temperature_enabled": (
        "Water temperature (simulation_settings.temperature). Needs the "
        "Daylight calendar on, which supplies the date.\n\n"
        "When on, the Resting Metabolism of every decision maker with a "
        "Metabolism Q10 becomes temperature dependent:\n"
        "  RM(t) = RM * Q10 ^ ((T(t) - t_ref) / 10)\n"
        "with T taken from the group's layer (see Layers and Group "
        "layers). This scales all action costs (rest, eat, move), so "
        "cold water makes ectotherms cheaper to run. Attack rates are "
        "not temperature-scaled. Adds no policy input: checkpoints stay "
        "valid."
    ),
    "temperature_group_layers": (
        "Which temperature layer each group lives in, as "
        "group=layer pairs separated by commas, e.g. "
        "'pelagic_fish=surface, gadoids=bottom'. The layer names must "
        "exist under Layers. Every decision maker with a Metabolism Q10 "
        "must be listed, otherwise loading the project fails."
    ),
    "temperature_layers": (
        "Temperature layers, one per line: 'name: 12 values' in deg C "
        "for Jan..Dec (each in -5..40), e.g.\n"
        "  surface: 4, 3, 3, 5, 9, 14, 17, 17, 14, 11, 8, 6\n"
        "Values are mid-month and interpolated circularly through the "
        "year. Groups are assigned to a layer under Group layers."
    ),

    # ------------------------------------------------------------------
    # Decision-maker FG editor
    # ------------------------------------------------------------------
    "max_energy_reserve": (
        "Max Energy Reserve ME (MJ per tonne of the group).\n\n"
        "The size of the energy store (fat/lipid) per tonne. The "
        "reserve fill s = R / (B * ME) drives the whole energy budget:\n"
        "- appetite h = min(1, max(0, (1 - s) / (1 - u))): full hunger "
        "up to the Maintenance Level u, zero when the store is full;\n"
        "- growth when s > u, starvation when s < u;\n"
        "- s is the 'own energy' input the policy observes.\n"
        "Energy eaten above B * ME is lost. Together with Energy "
        "Content it caps how fast mass can be built (ME / energy "
        "content tonnes per tonne). 0 = no store: permanent hunger and "
        "starvation."
    ),
    "energy_content": (
        "Energy Content (MJ per tonne of the group's own body).\n\n"
        "Two roles:\n"
        "- As prey: a predator gains energy content x assimilation "
        "factor per tonne it eats.\n"
        "- As the price of growth (mass balance, default on): building "
        "one new tonne costs this much from the reserve, and growth "
        "stops when the reserve would fall below u * ME * B. An "
        "income-funded group (Income growth kappa > 0) pays it out of "
        "this tick's intake instead, never out of the reserve.\n"
        "0 = worthless as food and growth paid by nothing (avoid). "
        "Library: zooplankton 4500, herring 4000 (lean; reserve on "
        "top), gadoids 5500, porpoises 8500."
    ),
    "resting_metabolism": (
        "Resting Metabolism RM (MJ per tonne per tick).\n\n"
        "The base metabolic cost. Every action pays a multiple of it, "
        "each tick, for the share of the biomass that chose it:\n"
        "  cost = B * share * RM * action cost * (1 + sum energy "
        "factors) * Q10 term\n"
        "with action cost 1 for Rest, Eat cost for eating and Move cost "
        "for moving. The cost is taken from the energy reserve (never "
        "below 0). A higher RM empties the reserve faster, so the group "
        "must eat more to stay above the Maintenance Level."
    ),
    "metabolism_q10": (
        "Metabolism Q10 (dimensionless; 0 or 1 = off). Only used when "
        "the project's Water temperature is on.\n\n"
        "Factor by which Resting Metabolism (and so every action cost) "
        "rises per 10 deg C warming:\n"
        "  RM(t) = RM * Q10 ^ ((T(t) - t_ref) / 10)\n"
        "Library: zooplankton 2.0, herring and gadoids 2.2; leave 0 for "
        "endotherms (porpoises, seals, birds). The group must also be listed under Group "
        "layers. Decision makers only."
    ),
    "metabolism_t_ref": (
        "Metabolism reference temperature t_ref (deg C), the water "
        "temperature at which the Resting Metabolism above applies "
        "unchanged.\n\n"
        "Empty = 'annual_mean': the Q10 factor is normalised to a "
        "yearly mean of exactly 1, so the library RM stays the annual "
        "average cost. Give a temperature when RM was measured at a "
        "known temperature (e.g. herring's summer value at 14 C). Used "
        "only with Metabolism Q10 > 0 and Water temperature on."
    ),
    "ingestion_q10": (
        "Ingestion Q10 (dimensionless; 0 or 1 = off). Only used when "
        "the project's Water temperature is on (section 181).\n\n"
        "Scales the group's WHOLE functional response on every prey - "
        "attack rate and ceiling 1/h together, so the half-saturation "
        "stays put:\n"
        "  intake(t) = intake * f(t),  f = Q10 ^ (T(t) / 10)\n"
        "normalised to a yearly mean of exactly 1, so the library attack "
        "rates and handling times stay annual means and the growth "
        "budget is unchanged. Library: zooplankton 2.8 (copepod rates, "
        "Hansen et al. 1997). The group must be listed under Group "
        "layers. Decision makers only; a grazing non decision maker "
        "uses Grazing Q10."
    ),
    "growth_q10": (
        "Growth Q10 (dimensionless; 0 or 1 = off). Only used when the "
        "project's Water temperature is on (section 183).\n\n"
        "Scales the rate at which a reserve above the maintenance line "
        "becomes biomass:\n"
        "  growth(t) = growth_rate * f(t),  f = Q10 ^ (T(t) / 10)\n"
        "normalised to a yearly mean of exactly 1, so the derived growth "
        "rate stays an annual mean. Development and growth are paced by "
        "temperature even on a full reserve (copepods ~3.0, Huntley & "
        "Lopez 1992). Starvation keeps the library rate. The group must "
        "be listed under Group layers. Decision makers only, and not "
        "with Income growth kappa."
    ),
    "income_growth_kappa": (
        "Income growth kappa (fraction, 0-1; 0 = the reserve-funded "
        "growth law). Decision makers only (section 184).\n\n"
        "Each tick the group grows out of what it ate this tick:\n"
        "  G = min(kappa x intake - costs,  B x g x (s - u) x energy content)\n"
        "and the rest of the intake goes to the reserve. With nothing to "
        "eat it does not grow; the reserve only pays the costs, so the "
        "fat built in summer is kept for the winter. The reserve-funded "
        "law (kappa 0) converts the reserve into biomass whenever it is "
        "above the maintenance line, eating or not. As in the DEBlipid "
        "and Wisconsin bioenergetics models (Martin et al. 2017). "
        "Library: herring 0.75, the value at which the ideal-conditions "
        "budget closes at today's s*. Set Feeding season with it. Not "
        "combined with Growth Q10; Starve Rate must be set."
    ),
    "regional_carrying_capacity": (
        "Regional carrying capacity K (t/km2; 0 or empty = off). Decision "
        "makers with r_max only (section 185).\n\n"
        "r_max is the r of a logistic stock, which only holds far below "
        "its capacity. Every tick the group loses the fraction\n"
        "  r_max / 1460 x B_region / K\n"
        "of its biomass and reserve in every cell, B_region its mean "
        "density over the grid. This loss is NOT in the derived Max "
        "Growth, so at ideal food the stock grows at "
        "r_max (1 - B/K) - fishing and settles below K instead of "
        "growing until its food is gone. It stands for recruitment "
        "compensation at the stock scale, so it is regional, not per "
        "cell. K is today's EFFECTIVE capacity (lowered by trawled "
        "nurseries and warming), not the historical ceiling. Library: "
        "gadoids 1.02, where r_max 0.37 and F 0.12 are stationary at "
        "today's 0.69 t/km2 (research_notes/Torskfiskarnas reglering)."
    ),
    "feeding_season_months": (
        "Feeding season (months per year; 0 or empty = 12). Only with "
        "Income growth kappa (section 184).\n\n"
        "An income-funded group can only grow while food is there, so "
        "its derived growth rate is paid over these months: "
        "g = gross / ((s* - u) x 1460) x 12 / months. It sizes the rate, "
        "it is not a switch. Needs r_max (a derived growth rate). "
        "Library: herring 5.5 (Apr - mid-Sep, Bachiller et al. 2018)."
    ),
    "maintenance_level": (
        "Maintenance Level u (fraction 0-1 of a full reserve).\n\n"
        "The reserve fill that divides growth from starvation:\n"
        "- s > u: biomass grows, dB = B * Max Growth * (s - u); for an "
        "income-funded group (Income growth kappa > 0) that is only the "
        "ceiling, the growth being paid from this tick's intake (none "
        "without food, whatever the reserve);\n"
        "- s < u: biomass shrinks, loss = B * Starve Rate * (u - s);\n"
        "- appetite is full while s <= u and falls linearly to 0 at "
        "s = 1;\n"
        "- reserve-funded growth never draws the reserve below u * ME * B.\n"
        "It also enters the derived growth rate through s* - u (the "
        "ideal reserve s*, section 160) and is the "
        "centre of the random starting fill in training. Missing = 0. "
        "A high u means a group that needs a fat store before it can "
        "reproduce."
    ),
    "r_max": (
        "Max Net Growth r_max (per year, from the literature): the "
        "population's maximum net growth rate at low density, after "
        "all losses.\n\n"
        "When > 0, Max Growth is DERIVED from it so the group can pay "
        "for its known mortality and still reach r_max:\n"
        "  g = (r_max + M1 + sum M2) / ((s* - u) * 1460)\n"
        "(x 12 / Feeding season for an income-funded group). "
        "M1 = Natural Mortality as an annual rate (only with "
        "--mortality on), sum M2 = Predation Mortality of the checked "
        "predators in the FG Interactions tab, 1460 = ticks per year "
        "at 6 h, s* = the reserve the group settles on under ideal "
        "conditions (see the s* - u term). 0 = set Max Growth by hand.\n"
        "It is a low-density rate: with Regional carrying capacity K the "
        "stock grows at r_max (1 - B/K) instead (section 185)."
    ),
    "growth_rate": (
        "Max Growth MG (fraction of biomass per tick).\n\n"
        "With a reserve fill s above the Maintenance Level u the cell "
        "grows by dB = B * MG * (s - u), i.e. at most MG * (1 - u) per "
        "tick, and each new tonne is paid from the reserve at the "
        "Energy Content (mass balance).\n"
        "Income-funded group (Income growth kappa > 0, section 184): "
        "B * MG * (s - u) is only the ceiling; the growth is "
        "min(kappa x intake - costs, that ceiling x Energy Content) of "
        "this tick, so with no food there is none and the reserve is "
        "kept.\n\n"
        "Read-only while r_max > 0: it is then derived from r_max, "
        "Natural Mortality and the checked Predation Mortality pairs. "
        "Edit those instead."
    ),
    "starve_rate": (
        "Starve Rate (fraction of biomass per tick).\n\n"
        "Catabolism when the reserve fill s is below the Maintenance "
        "Level u: loss = B * starve rate * (u - s) per tick, worst at "
        "s = 0 (starve rate * u). The reserve shrinks in proportion. "
        "This is how an underfed population declines. 0 = use Max "
        "Growth as the starvation rate (not allowed for an "
        "income-funded group, whose Max Growth is season-sized)."
    ),
    "visibility_floor": (
        "Visibility Floor (fraction 0-1; default for every predator).\n\n"
        "The Rest action doubles as hiding (sinking deep, schooling "
        "tight, hauling out). Of the resting share pi_rest the visible "
        "biomass is\n"
        "  visible = 1 - pi_rest * (1 - floor)\n"
        "so the floor is the part that stays detectable even when all "
        "of the group hides. Predators can only eat, and other groups "
        "only observe, the visible biomass.\n"
        "0 = hiding is perfect; 1 = hiding protects nothing. A pair "
        "value in the Visibility Floor matrix overrides it for one "
        "predator (e.g. echolocating porpoises)."
    ),
    "natural_mortality": (
        "Natural Mortality M1 (fraction of biomass per tick).\n\n"
        "Losses that are NOT predation by another group in the model: "
        "disease, old age, unmodelled predators. Applied each tick as "
        "B * (1 - M1), the reserve in proportion. Only active with "
        "train.py/inference.py --mortality on (default off) and scaled "
        "by --mortality_multiplier; with it on, M1 is also added to the "
        "derived Max Growth so the population can pay for it.\n"
        "Annual rate = -1460 * ln(1 - M1) at 6 h ticks."
    ),
    "m1_visual_share": (
        "M1 Visual Share (0-1; 0 = off). Part of Natural Mortality "
        "caused by visual hunters outside the model (e.g. birds, fish "
        "not modelled).\n\n"
        "That part scales with the visible biomass (hiding by Rest "
        "protects) and, with the Daylight calendar, with the light: "
        "high by day, low at night. Visual + Tactile share must be "
        "<= 1; the rest of M1 stays constant. Normalised so a group "
        "hiding the Hide Reference share loses exactly M1 on average."
    ),
    "m1_tactile_share": (
        "M1 Tactile Share (0-1; 0 = off). Part of Natural Mortality "
        "caused by non-visual hunters outside the model (e.g. jellyfish, "
        "filter feeders at depth).\n\n"
        "That part hits HIDDEN (resting) biomass Depth Risk Ratio times "
        "harder than visible biomass, so hiding is not free. Visual + "
        "Tactile share must be <= 1."
    ),
    "depth_risk_ratio": (
        "Depth Risk Ratio rho (>= 0; 0/empty = 1). For the tactile part "
        "of Natural Mortality: risk per tonne of hidden (resting) "
        "biomass relative to visible biomass. rho > 1 makes hiding at "
        "depth dangerous; rho = 1 makes the tactile part independent of "
        "behaviour. Used only with M1 Tactile Share > 0."
    ),
    "hide_reference": (
        "Hide Reference (0-1; 0/empty = 0.5). The resting (hidden) "
        "share at which the exposure-weighted Natural Mortality equals "
        "the M1 value entered above. A group hiding more than this "
        "loses less to visual and more to tactile hunters; one hiding "
        "less, the opposite. Used only with a visual or tactile share."
    ),
    "max_intake_rate": (
        "Max Intake Rate = the attack rate a (per tick).\n\n"
        "How fast a predator clears visible prey in its cell. Per "
        "prey, the intake per tonne of predator is the Holling "
        "functional response\n"
        "  f = a*B / (1 + a*h*B + w*B_pred)   (type II; type III uses "
        "B^2)\n"
        "B = visible prey (t/cell), h = Handling time of the pair, w = "
        "Interference. The CEILING is 1/h, not a; a sets how quickly it "
        "is reached (half-saturation at B = 1/(a*h)). Actual eating is "
        "further scaled by the share choosing Eat and by appetite. "
        "With the Daylight calendar a pair's Dark Ratio scales a "
        "through the day (annual mean = this value).\n"
        "A pair with handling time 0 eats linearly (a*B, no ceiling) "
        "when the project has any pair with h > 0 or a specialist, "
        "otherwise a flat a per tick."
    ),
    "interference": (
        "Interference w (1/tonne; 0 = off; Beddington-DeAngelis).\n\n"
        "Crowding among predators: w * (own biomass in the cell) is "
        "added to the denominator of every prey's functional response, "
        "so intake per tonne falls as the group packs together. This "
        "spreads predators out and prevents everyone from stacking on "
        "the same prey patch. 0 = pure Holling (no crowding)."
    ),
    "movement_speed": (
        "Movement Speed v (0-1; per tick).\n\n"
        "Of the biomass that chooses a move direction, the share v "
        "crosses into the neighbouring cell this tick and 1 - v stays. "
        "So v is the expected distance in cells per tick for the moving "
        "part; movement is always to one of the 4 neighbours. The Move "
        "cost is paid on all biomass that chose to move, whatever v. "
        "v = 0 removes the move actions entirely."
    ),
    "min_split_biomass": (
        "Indivisible Weight (kg): the mass of one individual (or the "
        "smallest unit that acts as one). 0 = continuous biomass.\n\n"
        "It keeps small populations behaving like animals, not "
        "fluid:\n"
        "- a cell with less than one unit acts as one: a single "
        "(argmax) action instead of a mix;\n"
        "- moves that would leave a destination below the extinction "
        "threshold are cancelled;\n"
        "- cells below Extinction Threshold x this weight die out;\n"
        "- at spawn every occupied cell gets at least 5 units."
    ),
    "extinction_threshold_factor": (
        "Extinction Threshold (multiple of the Indivisible Weight; "
        "default 0.5; 0 = off).\n\n"
        "At the end of each tick a cell holding more than 0 but less "
        "than factor x indivisible weight is set to zero, reserve "
        "included (booked as starvation). Prevents 'ghost' populations "
        "of a few grams spread over the grid. No effect when the "
        "Indivisible Weight is 0."
    ),
    "current_response": (
        "External Forces (0-1): the share of the water's drift "
        "(wind/currents) the group is carried by.\n\n"
        "Only active with --currents on. After the policy's own "
        "movement, a fraction (drift strength x 0.5 x this value) of "
        "the cell's biomass is moved downstream. 1 = passive drifter "
        "(plankton), 0 = holds position against the current (fish, "
        "mammals). An empty field means 0 for decision makers."
    ),
    "initial_biomass_range": (
        "Initial Total Biomass Range (tonnes over the WHOLE grid, not "
        "per cell). Training only.\n\n"
        "Each rollout (or each generation with a spawn seed) draws a "
        "total uniformly in [Min, Max] and spreads it over the grid "
        "with the Spawn Strategy below. A wide range trains the "
        "policies on many population sizes. Values are for the "
        "Reference Grid and scaled with the grid's area. Inference "
        "uses the fixed value in the Inference tab instead."
    ),
    "action_costs": (
        "Action Costs: multipliers of Resting Metabolism for the Eat and "
        "Move actions (Rest costs 1 x RM). Each tick, the share of the "
        "biomass choosing an action pays RM x its multiplier from the "
        "energy reserve. They set the energy trade-off the policy "
        "learns: a foraging trip has to pay for itself."
    ),
    "feeding_cost": (
        "Eat cost (multiple of Resting Metabolism; default 3).\n\n"
        "Energy per tonne per tick paid by the biomass that chooses an "
        "Eat action: RM x eat cost (x impact energy factors x Q10 "
        "term). Higher values make hunting expensive, so eating only "
        "pays where prey is dense. Must be >= the rest cost (1); the "
        "editor refuses a group that cannot break even at the "
        "Maintenance Level."
    ),
    "movement_cost": (
        "Move cost (multiple of Resting Metabolism; default 3).\n\n"
        "Energy per tonne per tick paid by the biomass that chooses a "
        "move direction: RM x move cost (x impact energy factors x Q10 "
        "term), whatever its Movement Speed. Higher values make "
        "searching expensive and favour staying put."
    ),

    # ------------------------------------------------------------------
    # Non-decision-maker FG editor
    # ------------------------------------------------------------------
    "ndm:r_max": (
        "Net Growth r_max (per year, from the literature): the "
        "producer's maximum net growth at low density.\n\n"
        "When > 0, Max Growth is DERIVED from it so the group can carry "
        "its own losses and its known grazing and predation:\n"
        "  r = (r_max + M1 + sum M2) / 1460\n"
        "M1 = Natural Mortality as a yearly rate (left out with "
        "--mortality off), "
        "sum M2 = Predation Mortality of the checked, active decision-"
        "maker predators in the FG Interactions tab, 1460 = ticks per "
        "year at 6 h. 0 = set Max Growth by hand."
    ),
    "ndm:natural_mortality": (
        "Natural Mortality M1 (fraction of the biomass lost per tick; "
        "0 = none).\n\n"
        "The producer's own losses that no group in the model eats: "
        "respiration, sinking out of the mixed layer, cell lysis. "
        "Removed every tick before growth, so in a dark winter, when "
        "light-limited growth is small, the stock declines. It enters "
        "the growth budget (r_max + M1 + sum M2), so the net growth at "
        "the light reference day is unchanged. Follows --mortality and "
        "--mortality_multiplier like a decision maker's M1. A nutrient "
        "pool is not given back (section 173)."
    ),
    "ndm:growth_rate": (
        "Max Growth r (logistic rate per tick).\n\n"
        "Each cell grows as dB = r * L(t) * B * (1 - B / K), K = Max "
        "Carrying Capacity, L(t) = light factor (1 unless the producer "
        "is light-limited). Then the Seed Rate is added and B is "
        "clipped to [0, K].\n\n"
        "Read-only while r_max > 0: it is then derived from r_max and "
        "the checked Predation Mortality pairs."
    ),
    "max_carrying_capacity": (
        "Max Carrying Capacity K (tonnes per cell).\n\n"
        "Both the logistic K (growth slows to 0 as B approaches K) and "
        "a hard ceiling on biomass per cell. It sets how much food the "
        "base of the web can offer per km2 and so, indirectly, how much "
        "consumer biomass the system can hold. Not scaled with the "
        "grid (it is already per cell)."
    ),
    "ndm:energy_content": (
        "Energy Content (MJ per tonne). What a predator gains per tonne "
        "eaten, times the pair's Assimilation Factor. Non decision "
        "makers have no energy reserve, so this is their only energy "
        "role: it sets their value as food."
    ),
    "seed_rate": (
        "Seed Rate (fraction of K per tick).\n\n"
        "Every cell receives seed rate x K x a random factor (0.1x-10x, "
        "log-uniform) each tick: spores, larvae and drift-in from "
        "outside. It is the only way back from a grazed-out cell, "
        "since zero biomass cannot grow logistically. 0 = no "
        "recolonisation; too high a value keeps every cell green "
        "whatever the grazing."
    ),
    "light_saturation": (
        "Light Saturation I_k (umol photons/m2/s; one value or 12 "
        "monthly values Jan..Dec; 0/empty = off).\n\n"
        "Makes growth light-limited on the Daylight calendar. "
        "Photosynthesis follows a P-I curve tanh(I / I_k), averaged "
        "over the mixed layer and the tick, so growth is zero at night "
        "and low in a deep or dark winter mixed layer. Max Growth is "
        "then multiplied by P(t) / P_ref (see Light Reference Day).\n"
        "Needs the Daylight calendar and the Light climate; ignored "
        "while Daylight is off. Library phytoplankton: 12 monthly values "
        "from 111 (Nov) to 298 (Jul)."
    ),
    "water_layer": (
        "Which layer of the water column the producer lives in: 'mixed' "
        "(the default, empty) or 'subsurface' (section 180).\n\n"
        "A subsurface producer sits between the mixed layer and the "
        "manifest's nutrients.subsurface_layer.bottom_m: its light comes "
        "through the mixed layer and decays below it at the layer's own "
        "attenuation, its nutrient pool is fed from the reservoir below "
        "and feeds the mixed layer's pool, and a deepening mixed layer "
        "entrains its water and algae (the autumn bloom). Grazers that "
        "feed on concentration dilute it over the layer thickness. Needs "
        "exactly one mixed and one subsurface producer, with one nutrient "
        "yield."),
    "nutrient_half_saturation": (
        "Nutrient Half-Saturation k_N (umol N/l; 0/empty = off).\n\n"
        "Gives the producer a dissolved nutrient pool per cell (section "
        "172). Growth becomes Max Growth x light x N / (N + k_N) x B and "
        "is paid out of the pool, instead of logistic growth toward the "
        "Carrying Capacity, which then only caps the stock. The pool is "
        "refilled from below the mixed layer (project: "
        "simulation_settings.nutrients) and by grazing (Grazing "
        "Regeneration). Ignored unless the project enables nutrients; "
        "needs the Daylight light climate (its mixed-layer depths)."
    ),
    "nutrient_yield": (
        "Nutrient Yield (ton/cell of producer per umol N/l in a 1 m water "
        "column).\n\n"
        "Converts the nutrient concentration to the biomass it can make: "
        "1 umol N/l over 1 m = 14 mg N/m2, times the Redfield C:N mass "
        "ratio and divided by the producer's carbon : wet-weight ratio. "
        "The pool is kept in the producer's own units (ton/cell), so this "
        "is the only stoichiometry in the model."
    ),
    "grazing_regeneration": (
        "Grazing Regeneration (0-1).\n\n"
        "Share of the producer biomass removed by grazers that returns to "
        "the nutrient pool at once (excretion, sloppy feeding, fast "
        "remineralisation in the mixed layer). The rest leaves the cycle "
        "with the grazers. It sets the regenerated, summer production."
    ),
    "stress_loss_max": (
        "Stress Loss Max (fraction of the biomass per tick at full "
        "nutrient exhaustion; 0/empty = off).\n\n"
        "Lysis and aggregate sinking once the nutrient pool runs dry "
        "(section 175): extra loss L_s x c(1 - f)/(f + c), f = N/(N + "
        "k_N) the growth's nutrient limitation. 0 when nutrients are "
        "replete, L_s when exhausted. Exported, not returned to the pool. "
        "Literature after nutrient exhaustion: 0.1-0.5 per day "
        "(0.025-0.125 per 6 h tick). Needs the nutrient pool."
    ),
    "stress_loss_shape": (
        "Stress Loss Shape c (0/empty = 0.1).\n\n"
        "How fast the stress loss rises as nutrients run out. 0.1 is "
        "ERSEM's diatom lysis curve (0.045 / 0.125 / 0.5 per day at f = "
        "1 / 0.3 / 0) without its replete baseline."
    ),
    "stress_regeneration": (
        "Stress Regeneration (0-1).\n\n"
        "Share of the nutrient-stress loss (lysis) that is remineralised "
        "in the mixed layer and returned to the nutrient pool; the rest "
        "sinks out (section 177)."
    ),
    "growth_efficiency": (
        "Growth Efficiency (0-1, grazing non decision makers).\n\n"
        "Tonnes of grazer grown per tonne of prey it takes (gross growth "
        "efficiency). Microzooplankton: 0.3 (Straile 1997). A non "
        "decision maker with a checked, attack-rate-carrying pair in "
        "the FG Interactions grazes in the population step (section 177)."
    ),
    "excretion_regeneration": (
        "Excretion Regeneration (0-1, grazing non decision makers).\n\n"
        "Share of the eaten prey not turned into growth that returns to "
        "the prey's nutrient pool as excreted ammonium. 0.571 x (1 - "
        "0.3) = 0.4 of the intake for microzooplankton."
    ),
    "grazing_q10": (
        "Grazing Q10 (grazing non decision makers; 0/empty = none).\n\n"
        "Temperature multiplier on the grazing rate, Q10^((T - T_ref)/10), "
        "T from the group's layer in the project's water temperature "
        "(needs the Daylight calendar). Microzooplankton: 2.6."
    ),
    "grazing_t_ref": (
        "Grazing Reference Temperature (deg C).\n\n"
        "The temperature the grazing pair's rates are given at "
        "(microzooplankton: 10 C)."
    ),
    "light_reference_day": (
        "Light Reference Day (day of year 1-365; 0/empty = 105, mid "
        "April): the day on which the light-limited growth factor is 1, "
        "i.e. on which Max Growth applies unchanged. Other days are "
        "scaled by their light relative to this day - the factor is not "
        "normalised over the year. Used only with Light Saturation > 0."
    ),
    "ndm:current_response": (
        "External Forces (0-1): the share of the water's drift "
        "(wind/currents) the group is carried by.\n\n"
        "Only active with --currents on. Each tick a fraction (drift "
        "strength x 0.5 x this value) of the cell's biomass is moved "
        "downstream. 1 = passive drifter (phytoplankton), 0 = attached "
        "(benthic community). An empty field means 1 for non decision "
        "makers."
    ),

    # ------------------------------------------------------------------
    # Interaction matrices (row = predator, column = prey)
    # ------------------------------------------------------------------
    "matrix:preys_on": (
        "Predation: checked = the row group eats the column group.\n\n"
        "Gives the predator an 'Eat <prey>' action (masked where the "
        "prey is absent), adds the pair to the functional response, "
        "makes the pair's Predation Mortality count in the prey's "
        "growth, and forces Observability on. A predator with exactly "
        "one prey is a specialist and gets a type III (sigmoid) "
        "response, a generalist type II. Changing the diet changes the "
        "action set: retraining is required."
    ),
    "matrix:predation_mortality": (
        "Predation Mortality M2 (per year, from the literature): the "
        "mortality the row predator causes in the column prey.\n\n"
        "It does NOT kill anything in the tick - actual predation comes "
        "from the functional response. It is added to the prey's "
        "derived Max Growth so the prey can sustain that pressure:\n"
        "  checking a predator raises the prey's growth, unchecking "
        "lowers it.\n"
        "Counts only while Predation is checked, the predator is active "
        "and the prey has r_max > 0."
    ),
    "matrix:observes": (
        "Observability: checked = the row group sees the column group.\n\n"
        "Adds 5 policy inputs: the column group's VISIBLE biomass in "
        "the own cell and in the 4 neighbours (hiding and the pair's "
        "Visibility Floor apply). A group always observes itself, and "
        "always observes its prey. Observing predators lets a policy "
        "learn to flee or hide. Any change alters the input size: "
        "existing checkpoints become invalid."
    ),
    "matrix:assimilation_factor": (
        "Assimilation Factor (0-1; default 1): share of the prey's "
        "Energy Content the predator actually absorbs. The prey biomass "
        "is removed in full; the rest is lost (faeces, inedible parts). "
        "Low values make a prey poor food: the predator must eat more "
        "of it to break even."
    ),
    "matrix:handling_time": (
        "Handling time h (ticks per tonne of prey per tonne of "
        "predator).\n\n"
        "Time spent catching and digesting. It sets the CEILING of the "
        "functional response: at most 1/h tonnes of prey per tonne of "
        "predator per tick, however dense the prey. With the attack "
        "rate a the half-saturation density is 1/(a*h) t/cell. "
        "The Auto button sets h = 1/a (ceiling = a). 0 = no ceiling: "
        "linear intake a*B when the project has any pair with h > 0 or "
        "a specialist, otherwise a flat a per tick."
    ),
    "matrix:visibility_floor": (
        "Visibility Floor for this pair (0-1; empty = use the prey's "
        "own Visibility Floor).\n\n"
        "The share of the prey that THIS predator still detects when "
        "the prey hides. Detection is a property of the pair: a "
        "porpoise's biosonar is not fooled by the darkness and depth "
        "that hide herring from seabirds. Limits both what the "
        "predator can eat and what it observes."
    ),
    "matrix:dark_ratio": (
        "Dark Ratio (0-1; empty or 1 = not light dependent): the row "
        "predator's attack rate in darkness relative to full daylight. "
        "Only with the Daylight calendar on.\n\n"
        "  a(t) = a * (rho + (1 - rho) * F(t)) / annual mean\n"
        "F = daylight detection (0 night, 1 day). The library attack "
        "rate stays the annual mean, so the pair hunts harder by day "
        "and less at night, and the season shifts the balance. 0.1 = "
        "strongly visual hunter."
    ),
    "matrix:light_threshold_deg": (
        "Light Threshold (sun elevation in degrees, -18..10; empty = "
        "-6, end of civil twilight): the sun elevation at which the "
        "predator's detection is half way between dark and light. A "
        "lower value = a predator that still sees well in twilight. "
        "Used only with a Dark Ratio < 1 and the Daylight calendar on."
    ),
    "matrix:impact_affects": (
        "Impact Affects: checked = the row group is affected by the "
        "column impact (fishing, noise, ...). The effect itself is "
        "defined by the pair's Impact Table; unchecked, or an empty "
        "table, means no effect. Every group can carry impacts; a non "
        "decision maker (e.g. bottom trawling on the benthos) feels only "
        "the biomass factor."
    ),
    "matrix:impact_table": (
        "Impact Table: how strongly the impact value in a cell affects "
        "the row group. Rows of (value, biomass factor, energy factor), "
        "linearly interpolated and clamped at the ends. Biomass factor "
        "= fraction of the cell's biomass lost per tick; energy factor "
        "= extra cost added to all action costs (decision makers only). "
        "Click to edit."
    ),

    # ------------------------------------------------------------------
    # Derived growth_rate panel (FG Interactions tab). Each term of a
    # row has its own key; the editor puts the FG's own numbers above.
    # ------------------------------------------------------------------
    "growth:panel": (
        "Derived growth_rate: every group's growth rate as computed from "
        "its literature budget (lib/world/growth_budget.py), live.\n\n"
        "Recomputed whenever a Predation cell, an M2 value, an "
        "assimilation factor or a handling time changes, or r_max, "
        "Natural Mortality, the Maintenance Level, Max Energy Reserve, "
        "Energy Content, Resting Metabolism, the Eat cost or Max Intake "
        "Rate is edited in the FG Editor (the last five through the "
        "ideal reserve s*), and Income growth kappa or Feeding season "
        "(the 12 / months factor). Applying an FG re-derives every group, since "
        "a prey's energy moves its predators' s*. Muted groups count as absent "
        "predators. The values are written to the library's "
        "growth_rate with 'Apply All Matrix Changes'. Hover over each "
        "term of a row for what it means."
    ),
    "growth:dm_formula": (
        "Decision maker budget:\n"
        "  g = (r_max + M1 + sum M2) / ((s* - u) x 1460)\n\n"
        "g is the group's Max Growth: per tick a cell grows by "
        "B x g x (s - u). Under ideal conditions (unlimited best prey, "
        "eating every tick) the reserve settles at s*, below a full "
        "reserve because appetite falls as the store fills, and over a "
        "year the growth there is (s* - u) x 1460 x g = r_max + M1 + "
        "sum M2, the GROSS growth. So a well-fed population at low "
        "density can pay its natural mortality (M1) and the predation "
        "pressure of its checked predators (M2) and still grow at its "
        "literature rate r_max. Hungrier populations grow less "
        "(section 160).\n\n"
        "Income-funded group (Income growth kappa > 0, section 184): "
        "g x 12 / Feeding season. It grows only out of what it eats, "
        "G = min(kappa x intake - costs, B x g x (s - u) x energy "
        "content), so the budget has to be produced in its feeding "
        "months; the food decides when."
    ),
    "growth:ndm_formula": (
        "Non decision maker budget:\n"
        "  r = (r_max + sum M2) / 1460\n\n"
        "r is the logistic growth rate per tick: at low density "
        "(B << K) and with full light the producer grows about r per "
        "tick, r_max + sum M2 per year. There is no M1 term: non "
        "decision makers have no natural mortality in the tick, so "
        "their r_max is already net of unmodelled losses. No s* - u "
        "factor either, as they have no energy reserve."
    ),
    "growth:growth_rate": (
        "g, the derived growth rate per tick at the library tick of "
        "6 h.\n\n"
        "Decision maker: Max Growth, dB = B x g x (s - u) per tick "
        "(for an income-funded group the ceiling of the growth it pays "
        "out of this tick's intake).\n"
        "Non decision maker: the logistic r.\n"
        "It replaces the hand-set Max Growth in the FG Editor, which is "
        "read-only while r_max > 0. A run with another --tick-length "
        "converts it in memory as (1 + g)^k - 1; the library keeps the "
        "6 h value."
    ),
    "growth:r_max": (
        "r_max (per year, FG Editor): the literature maximum NET "
        "population growth at low density - recruitment and individual "
        "growth with all mortality already removed. It is the part of "
        "the budget that is left as real increase once M1 and M2 are "
        "paid. Checking or unchecking predators never changes it; it "
        "is the invariant the budget is built around."
    ),
    "growth:m1": (
        "M1, Natural Mortality as an annual rate:\n"
        "  M1 = -1460 x ln(1 - natural_mortality per tick)\n\n"
        "Losses not modelled as predation by another group (disease, "
        "age, unmodelled predators); set in the FG Editor. Adding it to "
        "the budget lets the population replace those losses.\n"
        "Shown here always, but a run with --mortality off (the "
        "default) applies no M1 in the tick and leaves it out of g too: "
        "growth is then lower by this term's share."
    ),
    "growth:m2": (
        "M2, Predation Mortality of a checked predator on this group "
        "(per year; the Predation Mortality matrix in the FG "
        "Interactions tab).\n\n"
        "The literature mortality that predator causes. It is added to "
        "the prey's growth so the prey can sustain that pressure; it "
        "kills nothing by itself - real predation in the tick comes "
        "from the predator's functional response. Unchecking the pair, "
        "or muting the predator, removes the term. Counts only for "
        "decision-maker predators with M2 > 0."
    ),
    "growth:gate": (
        "s* - u: the energy surplus at the reserve s* this group settles "
        "on under ideal conditions, minus its Maintenance Level u (FG "
        "Editor).\n\n"
        "Growth per tick is g x (s - u) (its ceiling for an "
        "income-funded group). The hunger gate closes as the "
        "reserve fills, so a full reserve (s = 1) is never reached and "
        "dividing by 1 - u left every group short of its budget. s* is "
        "where an animal with unlimited prey of its best checked pair, "
        "eating every tick at the Holling ceiling, settles once the growth "
        "is paid from the reserve and dilutes it; dividing by s* - u makes "
        "the production there equal to the gross annual rate (section "
        "160). It depends on the predator's Max Energy Reserve, energy "
        "content, Resting Metabolism, feeding cost and the best prey's "
        "energy, assimilation and handling time. A budget above what that "
        "energy allows is an error, and so is a g above Max Energy "
        "Reserve / energy content (the mass balance caps it). A group "
        "without that data uses 1 - u."
    ),
    "growth:ticks_per_year": (
        "1460 = ticks per year at the library tick length "
        "(365 days x 24 h / 6 h). Converts the annual rates of the "
        "budget into a per-tick growth rate. The library is always "
        "calibrated at 6 h, whatever --tick-length a run uses."
    ),
    "growth:season": (
        "12 / Feeding season: an income-funded group (Income growth kappa "
        "> 0) grows only while it eats, so its annual budget has to be "
        "produced in its feeding months and g is raised by this factor "
        "(section 184). The months size the rate; when the group grows "
        "is decided by its food. s* is unchanged."
    ),
    "growth:hand_set": (
        "This group has no r_max (or r_max = 0), so its growth rate is "
        "the value typed as Max Growth in the FG Editor. It does not "
        "follow the predation matrix: checking a predator does not "
        "raise it. Set r_max > 0 to derive it from the budget."
    ),
    "growth:library_note": (
        "The library (fg_library.yaml) still holds this older growth "
        "rate. The derived value in front of it is what the current "
        "matrix gives; it is written to the library with 'Apply All "
        "Matrix Changes' (or Apply Changes in the FG Editor). Until "
        "then, training uses the library value."
    ),
    "growth:error": (
        "The budget cannot be computed: the Maintenance Level must be "
        "below 1 (otherwise there is no growth window) and Natural "
        "Mortality below 1 per tick. Fix the value in the FG Editor."
    ),

    # ------------------------------------------------------------------
    # Impact editor and impact tables
    # ------------------------------------------------------------------
    "impact_value_range": (
        "Value Range (Min, Max; in the impact's unit). Training only.\n\n"
        "The range of the random impact field each rollout. Without a "
        "spawn shape every cell is uniform in [Min, Max]; with one, the "
        "pattern is stretched to [Min, Max]. Min = Max gives a constant "
        "field. Training the policies over the whole range of "
        "intensities they may meet lets them respond to any project "
        "scenario. Inference ignores it and uses the .npz maps. Also "
        "sets the allowed values in the Impact Tables."
    ),
    "impact_observable": (
        "Observable by policy: the decision makers see this impact.\n\n"
        "On: adds 5 inputs per policy (the impact value in the own cell "
        "and 4 neighbours), so groups can learn to avoid it; the field "
        "is spawned with the Spawn Strategy below.\n"
        "Off: the impact is zero during training, but in inference it "
        "still acts on mortality and costs while the policies cannot "
        "see it.\n"
        "Changing it alters the input size: retraining is required."
    ),
    "impact_baseline_map": (
        "Baseline map: today's field of this impact as a .npz, one value "
        "per cell (e.g. fishing effort in the cell).\n\n"
        "Training: every training world gets this map, whether or not the "
        "policies can observe the impact, so they train in today's world; "
        "it replaces the spawned field of an observable impact.\n"
        "Inference: not read. Add the same file under the Inference tab "
        "to keep today's field in an inference world, or another file "
        "for a scenario (closed wind farm, changed quotas) - one field "
        "per impact, never added on top.\n"
        "No baseline map: absent from training unless observable. Adds "
        "no inputs; checkpoints stay valid."
    ),
    "impact_table_value": (
        "Impact value (in the impact's unit) at which this row's "
        "factors apply. Rows are sorted by value and interpolated "
        "linearly; below the first and above the last row the end "
        "values apply. Limited to the impact's Value Range."
    ),
    "impact_table_biomass_factor": (
        "Biomass factor (0-1): FRACTION OF THE CELL'S BIOMASS LOST PER "
        "TICK at this impact value (not the surviving fraction). Applied "
        "before predation, reserve in proportion; the losses of several "
        "impacts add up. Small numbers matter: 0.001 per 6 h tick, "
        "sustained, removes about 77 % of a population in a year."
    ),
    "impact_table_energy_factor": (
        "Energy factor (0-1): EXTRA METABOLIC COST at "
        "this impact value. Added to the cost multiplier of every "
        "action, (1 + sum of energy factors) x the normal Rest/Eat/Move "
        "cost - e.g. stress from noise. 0.5 = all action costs 50 % "
        "higher in that cell. It does not remove reserve directly."
    ),

    # ------------------------------------------------------------------
    # Spawn strategy
    # ------------------------------------------------------------------
    "spawn_mode": (
        "Spawn mode: how the group's starting biomass is spread over "
        "the grid at the start of each rollout (training and "
        "inference).\n"
        "Uniform = evenly; Perlin = smooth patches; Colony = a few "
        "aggregations; Env-driven = follows other groups' fields.\n"
        "Every occupied cell gets at least 5 x the Indivisible Weight; "
        "the total stays exactly the drawn biomass. See the Info panel "
        "for the selected mode."
    ),
    "impact_spawn_mode": (
        "Spawn mode of this impact's random training field. The shape "
        "is stretched to the Value Range (Colony jitter keeps absolute "
        "levels). Env-driven has no other fields to follow for "
        "impacts: it gives a constant field at Min unless noise is "
        "added. See the Info panel for the selected mode."
    ),
    "spawn_templates": (
        "Saved spawn settings for this group (or impact) and the "
        "current mode. Pick one to load it; Save stores the current "
        "parameters under a name, Delete removes the selected one. "
        "Stored in the project file."
    ),
    "spawn_scale": (
        "Scale (cells): width of the smoothing of the base octave, "
        "i.e. the typical patch size. Larger = fewer, bigger patches. "
        "Not scaled with the grid size."
    ),
    "spawn_octaves": (
        "Octaves (1-6): number of noise layers. Each further octave "
        "adds finer detail (scale / lacunarity) on top of the base "
        "patches."
    ),
    "spawn_persistence": (
        "Persistence (0-1): amplitude of each octave relative to the "
        "previous one. Low = smooth large patches; high = rough, "
        "detailed field."
    ),
    "spawn_lacunarity": (
        "Lacunarity (~2): factor by which the patch size shrinks from "
        "one octave to the next."
    ),
    "spawn_threshold": (
        "Threshold (0-1): subtracted from the normalised field; cells "
        "that fall to 0 or below get no biomass. Higher = fewer, "
        "sharper-edged patches with empty water between."
    ),
    "spawn_min_frac_of_max": (
        "Min frac of max (0-1; 0 = off): every non-empty cell is lifted "
        "to at least this fraction of the strongest cell. Cells already "
        "zeroed stay empty. High values (> 0.5) flatten the field "
        "towards uniform."
    ),
    "spawn_n_colonies": (
        "N colonies: number of aggregation centres, placed at random "
        "allowed cells. Scaled with the grid's area relative to the "
        "Reference Grid. Few colonies suit apex predators with small, "
        "clumped populations."
    ),
    "spawn_sigma_cells": (
        "Sigma (cells): radius (Gaussian sigma) of each colony. Small = "
        "tight clusters; large = diffuse aggregations."
    ),
    "spawn_anchor": (
        "Anchor (free / coast / open_water): where colony centres may "
        "be placed. Reserved for when a depth map is wired in - "
        "currently every choice places centres uniformly."
    ),
    "spawn_amplitude_mode": (
        "Amplitude mode.\n"
        "uniform: all colonies equally strong, summed.\n"
        "jitter: each colony gets its own random strength in "
        "[amplitude min, amplitude max], colonies combine by maximum. "
        "Mainly for impacts: sources of different strength (e.g. noise "
        "at different levels) keep their absolute levels. For biomass "
        "only the relative shape changes."
    ),
    "spawn_amplitude_min": (
        "Amplitude min (0-1): lower bound of each colony's random "
        "strength in jitter mode, relative to the range (for impacts: "
        "Min + (Max - Min) x amplitude)."
    ),
    "spawn_amplitude_max": (
        "Amplitude max (0-1): upper bound of each colony's random "
        "strength in jitter mode, relative to the range."
    ),
    "spawn_floor": (
        "Floor: subtracted from the combined weight; cells that fall to "
        "0 or below get no biomass. Acts as a threshold - higher = "
        "only the strongest areas of the referenced fields are "
        "occupied."
    ),
    "spawn_noise_amp": (
        "Noise amplitude (0 = none): strength of a smooth random field "
        "added to the weights, so the group follows its references "
        "loosely instead of copying them."
    ),
    "spawn_noise_scale": (
        "Noise scale (cells): patch size of the added noise field."
    ),
    "spawn_refs": (
        "Refs: other groups' freshly spawned biomass fields this group "
        "follows. Weight = sum over refs of weight x transform(field), "
        "e.g. zooplankton following phytoplankton. Referenced groups "
        "are spawned first; refs to unknown, muted or later groups are "
        "skipped."
    ),
    "spawn_ref_name": (
        "Ref name: id of the group whose spawned biomass field is used "
        "(e.g. phytoplankton)."
    ),
    "spawn_ref_weight": (
        "Ref weight: signed multiplier of the transformed field. "
        "Positive = attracted to it, negative = avoids it."
    ),
    "spawn_ref_transform": (
        "Ref transform applied to the field before weighting:\n"
        "linear = as is; exp = strongly peaked on the densest cells; "
        "invert = 1 - field/max (high where the reference is low); "
        "gauss_smooth = blurred, for a looser association."
    ),

    # ------------------------------------------------------------------
    # Inference tab
    # ------------------------------------------------------------------
    "inference_initial_biomass": (
        "Initial Biomass (tonnes over the whole grid) for inference.py "
        "and the training probe rollout. Replaces the training range: "
        "no random draw, so scenarios ('noll' vs 'projekt') start from "
        "the same state. Spread with the group's Spawn Strategy, energy "
        "reserve at 70 % of full. Scaled with the grid area like the "
        "training range. Empty = 0: the group starts absent."
    ),
    "inference_impact_map": (
        "Impact map (.npz) for inference: a 2-D array (the array named "
        "after the impact, else the first one) with the impact value "
        "per cell, resampled to the grid. Used as is - not clipped to "
        "the Value Range - for every active impact, observable or not. "
        "No map = a zero field. Drop a file here or click to browse."
    ),
}


def ndm_help_key(key):
    """The help key of a non-decision-maker field (``ndm:`` variant if any)."""
    nkey = f"ndm:{key}"
    return nkey if nkey in FIELD_HELP else key
