import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import codecs
try:
    from ruamel.yaml import YAML
except ImportError:
    import sys
    print("Error: The 'ruamel.yaml' library is required.")
    print("Please install it using: pip install ruamel.yaml")
    sys.exit(1)

# Initialize YAML handler
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

class FGConfigApp:
    def __init__(self, root):
        self.root = root
        self._base_title = "Ecosystem FG Configuration Tool"
        self.root.title(self._base_title)
        self.root.geometry("1000x700")
        # Dirty-tracking: True when in-memory project_data has unsaved
        # changes vs. the YAML file on disk. The title bar gets a leading
        # '*' marker, and closing the window prompts to save.
        self._dirty = False

        self.library_path = os.path.join(os.path.dirname(__file__), "fg_library.yaml")
        self.recent_path = os.path.join(os.path.dirname(__file__), "recent_projects.txt")
        self.max_recent = 5
        self.recent_projects = self.load_recent_projects()
        self.project_path = None
        
        self.global_library = self.load_yaml(self.library_path) or {"species_definitions": {}, "interaction_definitions": {}}
        # Ensure default impact_definitions exist in the global library
        if "impact_definitions" not in self.global_library or not self.global_library.get("impact_definitions"):
            self.global_library["impact_definitions"] = {
                "bottom_trawling":  {"display_name": "Bottom Trawling"},
                "pelagic_trawling": {"display_name": "Pelagic Trawling"},
                "hunting":          {"display_name": "Hunting"},
                "logging":          {"display_name": "Logging"},
                "chemicals":        {"display_name": "Chemicals"},
                "windfarm_noise":   {"display_name": "Windfarm Noise"},
                "ship_traffic":     {"display_name": "Ship Traffic"},
                "turbidity":        {"display_name": "Turbidity"},
            }
            self.save_yaml(self.global_library, self.library_path)
        self.project_data = {
            "project_metadata": {"name": "New Project"},
            "simulation_settings": {},
            "decision_makers": [],
            "non_decision_makers": [],
            "impact_variables": []
        }
        self.current_fg_configs = {} # local project configs for active FGs

        # Swedish name mapping for display (as requested)
        self.sv_mapping = {
            "phytoplankton": "Växtplankton",
            "zooplankton": "Djurplankton",
            "benthic_community": "Bottensamhälle",
            "pelagic_fish": "Pelagiska fiskar",
            "gadoids": "Torskfiskar",
            "porpoises": "Tumlare",
            "seals": "Sälar",
            "seabirds": "Sjöfåglar"
        }

        self.setup_ui()

        # Hook window close (X button) to the save-prompt flow.
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        # Tk-variable traces so editing the project name or reference grid
        # in the header row marks the project as dirty (these fields are
        # only persisted via save_project, but unsaved edits should still
        # be flagged).
        try:
            self.project_name_var.trace_add("write", lambda *_: self._mark_dirty())
            self.ref_grid_w_var.trace_add("write", lambda *_: self._mark_dirty())
            self.ref_grid_h_var.trace_add("write", lambda *_: self._mark_dirty())
        except Exception:
            pass

    def _on_mousewheel(self, event):
        """Handle mouse wheel and trackpad scroll events (vertical)."""
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta:
            delta = int(-1 * (event.delta / 120))
        else:
            return

        try:
            active_idx = self.notebook.index(self.notebook.select())
        except Exception:
            return
        if active_idx == 0 and hasattr(self, 'project_canvas') and self.project_canvas.winfo_exists():
            if self._canvas_is_scrollable(self.project_canvas, axis="y"):
                self.project_canvas.yview_scroll(delta, "units")
        elif active_idx == 1 and hasattr(self, 'matrix_canvas') and self.matrix_canvas.winfo_exists():
            if self._canvas_is_scrollable(self.matrix_canvas, axis="y"):
                self.matrix_canvas.yview_scroll(delta, "units")

    def _on_shift_mousewheel(self, event):
        """Handle horizontal scroll via Shift+wheel or trackpad horizontal gesture."""
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta:
            delta = int(-1 * (event.delta / 120))
        else:
            return

        try:
            active_idx = self.notebook.index(self.notebook.select())
        except Exception:
            return
        if active_idx == 0 and hasattr(self, 'project_canvas') and self.project_canvas.winfo_exists():
            if self._canvas_is_scrollable(self.project_canvas, axis="x"):
                self.project_canvas.xview_scroll(delta, "units")
        elif active_idx == 1 and hasattr(self, 'matrix_canvas') and self.matrix_canvas.winfo_exists():
            if self._canvas_is_scrollable(self.matrix_canvas, axis="x"):
                self.matrix_canvas.xview_scroll(delta, "units")

    @staticmethod
    def _canvas_is_scrollable(canvas, axis="y"):
        """Return True only if content size exceeds the visible canvas size along the given axis."""
        try:
            bbox = canvas.bbox("all")
            if not bbox:
                return False
            if axis == "y":
                return (bbox[3] - bbox[1]) > canvas.winfo_height()
            else:
                return (bbox[2] - bbox[0]) > canvas.winfo_width()
        except Exception:
            return False

    def load_yaml(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                content = f.read()
                if content.startswith(codecs.BOM_UTF8):
                    content = content[len(codecs.BOM_UTF8):]
                return yaml.load(content.decode('utf-8'))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def save_yaml(self, data, path):
        try:
            with open(path, 'wb') as f:
                f.write(codecs.BOM_UTF8)
                yaml.dump(data, f)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save to {path}: {e}")

    # ------------------------------------------------------------------
    # Dirty-tracking helpers. The window title reflects the dirty state:
    #   "* <base> — <project_basename>"   when there are unsaved changes
    #   "<base> — <project_basename>"     when clean
    # When closing the window (X button or File>Exit) we prompt the user
    # to save / discard / cancel via _on_close.
    # ------------------------------------------------------------------
    def _update_title(self):
        title = self._base_title
        if getattr(self, 'project_path', None):
            title = f"{title} — {os.path.basename(self.project_path)}"
        if getattr(self, '_dirty', False):
            title = "* " + title
        try:
            self.root.title(title)
        except Exception:
            pass

    def _mark_dirty(self, *_args):
        if not getattr(self, '_dirty', False):
            self._dirty = True
            self._update_title()

    def _clear_dirty(self):
        if getattr(self, '_dirty', False):
            self._dirty = False
        self._update_title()

    def _on_close(self):
        """Window close / Exit handler. Prompts to save unsaved changes."""
        if not getattr(self, '_dirty', False):
            self.root.destroy()
            return
        resp = messagebox.askyesnocancel(
            "Unsaved Changes",
            "You have unsaved changes in this project.\n\n"
            "Do you want to save them before exiting?",
        )
        if resp is None:
            # Cancel — don't close.
            return
        if resp:
            saved = self.save_project()
            if not saved:
                # User cancelled the file dialog — abort close.
                return
        self.root.destroy()

    def setup_ui(self):
        # Menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Project", command=self.new_project)
        filemenu.add_command(label="Open Project", command=self.open_project)
        filemenu.add_command(label="Save Project", command=self.save_project)
        self.recent_menu = tk.Menu(filemenu, tearoff=0)
        filemenu.add_cascade(label="Open Recent", menu=self.recent_menu)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
        self.refresh_recent_menu()

        # Main Layout
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=1, fill="both")

        # Tab 1: Project & FGs
        self.project_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.project_tab, text="Project & FGs")
        self.setup_project_tab()

        # Tab 2: Impacts (impact variables list + per-impact editor)
        self.impacts_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.impacts_tab, text="Impacts")
        self.setup_impacts_tab()

        # Tab 3: FG Interactions
        self.matrix_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_tab, text="FG Interactions")

        # Tab 4: Impact Interactions
        self.impact_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.impact_tab, text="Impact Interactions")

        # Tab 5: Inference
        self.inference_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.inference_tab, text="Inference")
        self.setup_inference_tab()

        self.setup_matrix_tab()

    def setup_impacts_tab(self):
        # Dedicated tab hosting the "Impact Variables in Project" list and
        # the per-impact editor (value range, observable flag, spawn-strategy).
        # Layout mirrors the structure these widgets used to have inside the
        # Project tab before they were moved here.
        parent = self.impacts_tab

        # Impact Variables list in project
        impact_frame = ttk.LabelFrame(parent, text="Impact Variables in Project")
        impact_frame.pack(expand=False, fill="x", padx=10, pady=5)

        self.impact_listbox = tk.Listbox(impact_frame, exportselection=False, height=8)
        self.impact_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        impact_btn_frame = ttk.Frame(impact_frame)
        impact_btn_frame.pack(side="right", fill="y", padx=5, pady=5)

        ttk.Button(impact_btn_frame, text="Add from Library", command=self.add_impact_from_library).pack(fill="x", pady=2)
        ttk.Button(impact_btn_frame, text="Remove Impact", command=self.remove_impact).pack(fill="x", pady=2)
        self.impact_mute_btn = ttk.Button(impact_btn_frame, text="Mute",
                                          command=self.toggle_mute_impact)
        self.impact_mute_btn.pack(fill="x", pady=2)
        self.impact_listbox.bind("<<ListboxSelect>>",
                                 lambda e: (self._refresh_mute_button_labels(),
                                            self.on_impact_select()))

        # Impact Editor: per-impact value range used to sample the impact map
        # uniformly per cell at training/inference time (replaces the old
        # PNG-based maps and zero dummies).
        self.impact_editor_frame = ttk.LabelFrame(parent, text="Impact Editor")
        self.impact_editor_frame.pack(fill="x", padx=10, pady=5)

        self.impact_editor_label_var = tk.StringVar(value="(no impact selected)")
        ttk.Label(self.impact_editor_frame, textvariable=self.impact_editor_label_var,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, columnspan=2,
                                                          sticky="w", padx=5, pady=(5, 2))

        ttk.Label(self.impact_editor_frame, text="Value Range").grid(
            row=1, column=0, sticky="w", padx=5, pady=2)
        ie_min_var, ie_max_var = self._build_value_range_row(
            self.impact_editor_frame, row=1)
        self.impact_value_min_var = ie_min_var
        self.impact_value_max_var = ie_max_var

        ttk.Label(self.impact_editor_frame, text="Observable by policy").grid(
            row=2, column=0, sticky="w", padx=5, pady=2)
        self.impact_observable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.impact_editor_frame,
                        variable=self.impact_observable_var,
                        command=self._on_impact_observable_toggled).grid(
            row=2, column=1, sticky="w", padx=5, pady=2)

        self.impact_apply_btn = ttk.Button(self.impact_editor_frame,
                                           text="Apply Changes",
                                           command=self.apply_impact_changes)
        self.impact_apply_btn.grid(row=3, column=0, columnspan=2, pady=5)

        # Spawn-strategy editor for impact maps. Mirrors the per-FG spawn
        # editor (mode/parameters/preview/info) but writes back to the
        # impact_variables[i].spawn block. Used at training time to shape
        # the random impact field (still scaled to [value_min, value_max]).
        self.impact_spawn_frame = ttk.LabelFrame(
            self.impact_editor_frame, text="Spawn Strategy")
        self.impact_spawn_frame.grid(row=4, column=0, columnspan=2,
                                     sticky="ew", padx=5, pady=5)
        self.impact_spawn_vars = {}
        self.impact_spawn_vars["_owner_kind"] = "impact"
        self._build_spawn_editor(self.impact_spawn_frame, self.impact_spawn_vars)
        # Hidden by default; shown only for observable impacts. The Spawn
        # Strategy only governs how observable impact maps are populated;
        # non-observable impacts are zero-filled at training time, so the
        # editor is meaningless for them.
        self.impact_spawn_frame.grid_remove()

        # Track widgets so we can enable/disable the editor based on selection
        # and muted status.
        self._impact_editor_widgets = list(self.impact_editor_frame.winfo_children())
        self._set_impact_editor_enabled(False)

    def setup_project_tab(self):
        # Scrollable container for the whole project tab
        self.project_canvas = tk.Canvas(self.project_tab, highlightthickness=0)
        self.project_scrollbar = ttk.Scrollbar(self.project_tab, orient="vertical", command=self.project_canvas.yview)
        self.project_inner = ttk.Frame(self.project_canvas)

        self.project_inner.bind(
            "<Configure>",
            lambda e: self.project_canvas.configure(scrollregion=self.project_canvas.bbox("all"))
        )
        inner_window = self.project_canvas.create_window((0, 0), window=self.project_inner, anchor="nw")
        self.project_hscrollbar = ttk.Scrollbar(self.project_tab, orient="horizontal", command=self.project_canvas.xview)
        self.project_canvas.configure(
            yscrollcommand=self.project_scrollbar.set,
            xscrollcommand=self.project_hscrollbar.set,
        )

        # Expand inner frame to canvas width only when it would otherwise be narrower,
        # so horizontal scrolling kicks in when content is wider than the canvas.
        def _on_canvas_configure(event, win=inner_window, cv=self.project_canvas, inner=self.project_inner):
            req_w = inner.winfo_reqwidth()
            cv.itemconfigure(win, width=max(event.width, req_w))
        self.project_canvas.bind("<Configure>", _on_canvas_configure)

        self.project_hscrollbar.pack(side="bottom", fill="x")
        self.project_scrollbar.pack(side="right", fill="y")
        self.project_canvas.pack(side="left", expand=True, fill="both")

        # Project Info
        info_frame = ttk.LabelFrame(self.project_inner, text="Project Info")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Layout: rad 0 = Project Name, rad 1 = Reference Grid.
        # Båda raderna delar samma kolumn-struktur i info_frame så att
        # Entry-fälten linjerar vertikalt:
        #   col 0: huvud-label ("Project Name:" / "Reference Grid ...:")
        #   col 1: "X:"-prefix (tom på namn-raden)
        #   col 2: första Entry (Project Name / X-värde)
        #   col 3: "Y:"-prefix (endast rad 1)
        #   col 4: andra Entry (Y-värde, endast rad 1)
        #   col 5: status-label (endast rad 1)
        ttk.Label(info_frame, text="Project Name:").grid(row=0, column=0, sticky="w", padx=5)
        self.project_name_var = tk.StringVar(value="New Project")
        # Project Name-Entry spänner över kolumnerna 2..4 så att fältet täcker
        # bägge X/Y-Entry-fälten nedan (inkl. "Y:"-prefix-label) och slutar
        # vid samma högerkant som Y-Entry. sticky="ew" gör att Entry:n växer
        # till hela columnspan-bredden.
        ttk.Entry(info_frame, textvariable=self.project_name_var).grid(
            row=0, column=2, columnspan=3, sticky="ew", padx=(0, 8))

        # Reference grid size. All biomass values entered in the FG editor
        # ("Initial Total Biomass Range") and the Inference tab
        # ("Initial Biomass") are interpreted as if the simulation runs on a
        # grid of exactly reference_grid_width x reference_grid_height cells.
        # When train.py / inference.py are run on a grid of a different size,
        # the runtime (`load_project_config`) scales those values linearly by
        # (actual_cells / reference_cells), with a hard floor of 1 ton on the
        # scaled lower bound and on the scaled inference initial biomass.
        ttk.Label(info_frame, text="Reference Grid (cells, X x Y, min 3):").grid(
            row=1, column=0, sticky="w", padx=5, pady=(4, 0))
        self.ref_grid_w_var = tk.StringVar(value="60")
        self.ref_grid_h_var = tk.StringVar(value="60")

        # Live validation: only positive-integer input is accepted on keypress
        # (empty allowed as a transient state during editing). A separate
        # visual cue (red background + status label) flags values < 3 in real
        # time without blocking the typing — the user must still type "30"
        # via "3" then "0", and we cannot block "1" or "2" outright because
        # those are valid prefixes of "10", "20", etc. Clamping to >= 3
        # happens at save_project() and at load_project_from_path().
        MIN_REF = 3
        self.ref_grid_min = MIN_REF

        def _pos_int(s):
            return s == "" or (s.isdigit() and (s == "0" or not s.startswith("0")))
        vcmd_ref = (info_frame.register(_pos_int), "%P")
        ttk.Label(info_frame, text="X:").grid(row=1, column=1, sticky="w", padx=(0, 2), pady=(4, 0))
        self.ref_grid_w_entry = ttk.Entry(
            info_frame, textvariable=self.ref_grid_w_var, width=10,
            validate="key", validatecommand=vcmd_ref)
        self.ref_grid_w_entry.grid(row=1, column=2, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Label(info_frame, text="Y:").grid(row=1, column=3, sticky="w", padx=(0, 2), pady=(4, 0))
        self.ref_grid_h_entry = ttk.Entry(
            info_frame, textvariable=self.ref_grid_h_var, width=10,
            validate="key", validatecommand=vcmd_ref)
        self.ref_grid_h_entry.grid(row=1, column=4, sticky="w", padx=(0, 8), pady=(4, 0))

        # Status label that turns red when either field is < 3 (or empty).
        self.ref_grid_status_var = tk.StringVar(value="")
        self.ref_grid_status_lbl = tk.Label(
            info_frame, textvariable=self.ref_grid_status_var,
            fg="red", font=("TkDefaultFont", 9, "italic"))
        self.ref_grid_status_lbl.grid(row=1, column=5, sticky="w", padx=(8, 0), pady=(4, 0))

        # tk.Entry supports a 'background' option that ttk.Entry does not.
        # We toggle a ttk style instead.
        try:
            _style = ttk.Style()
            _style.configure("Invalid.TEntry", fieldbackground="#ffd6d6")
        except Exception:
            pass

        def _validate_ref_live(*_args):
            def _state(val):
                if val == "":
                    return "empty"
                try:
                    iv = int(val)
                except ValueError:
                    return "bad"
                return "ok" if iv >= MIN_REF else "low"
            sw = _state(self.ref_grid_w_var.get())
            sh = _state(self.ref_grid_h_var.get())
            self.ref_grid_w_entry.configure(
                style="Invalid.TEntry" if sw != "ok" else "TEntry")
            self.ref_grid_h_entry.configure(
                style="Invalid.TEntry" if sh != "ok" else "TEntry")
            msgs = []
            if sw == "empty" or sh == "empty":
                msgs.append("ange värde")
            if sw == "low" or sh == "low":
                msgs.append(f"min {MIN_REF}×{MIN_REF}")
            self.ref_grid_status_var.set("; ".join(msgs))

        self._validate_ref_live = _validate_ref_live
        self.ref_grid_w_var.trace_add("write", _validate_ref_live)
        self.ref_grid_h_var.trace_add("write", _validate_ref_live)
        _validate_ref_live()

        # Two side-by-side FG frames: Decision Makers and Non Decision Makers
        fg_container = ttk.Frame(self.project_inner)
        fg_container.pack(expand=True, fill="both", padx=10, pady=5)

        # --- Decision Makers ---
        dm_frame = ttk.LabelFrame(fg_container, text="Decision Makers")
        dm_frame.pack(side="left", expand=True, fill="both", padx=(0, 5))

        self.fg_listbox = tk.Listbox(dm_frame, exportselection=False)
        self.fg_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.fg_listbox.bind("<<ListboxSelect>>",
                             lambda e: self.on_fg_select("decision_makers"))

        dm_btn_frame = ttk.Frame(dm_frame)
        dm_btn_frame.pack(side="right", fill="y", padx=5, pady=5)
        ttk.Button(dm_btn_frame, text="Add from Library",
                   command=lambda: self.add_from_library("decision_makers")).pack(fill="x", pady=2)
        ttk.Button(dm_btn_frame, text="Add New FG",
                   command=lambda: self.add_new_fg("decision_makers")).pack(fill="x", pady=2)
        ttk.Button(dm_btn_frame, text="Remove FG",
                   command=lambda: self.remove_fg("decision_makers")).pack(fill="x", pady=2)
        # Dynamic Mute/Unmute button: toggles the muted flag on the selected FG.
        self.dm_mute_btn = ttk.Button(dm_btn_frame, text="Mute",
                                      command=lambda: self.toggle_mute_fg("decision_makers"))
        self.dm_mute_btn.pack(fill="x", pady=2)

        # --- Non Decision Makers ---
        ndm_frame = ttk.LabelFrame(fg_container, text="Non Decision Makers")
        ndm_frame.pack(side="left", expand=True, fill="both", padx=(5, 0))

        self.ndm_listbox = tk.Listbox(ndm_frame, exportselection=False)
        self.ndm_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.ndm_listbox.bind("<<ListboxSelect>>",
                              lambda e: self.on_fg_select("non_decision_makers"))

        ndm_btn_frame = ttk.Frame(ndm_frame)
        ndm_btn_frame.pack(side="right", fill="y", padx=5, pady=5)
        ttk.Button(ndm_btn_frame, text="Add from Library",
                   command=lambda: self.add_from_library("non_decision_makers")).pack(fill="x", pady=2)
        ttk.Button(ndm_btn_frame, text="Add New FG",
                   command=lambda: self.add_new_fg("non_decision_makers")).pack(fill="x", pady=2)
        ttk.Button(ndm_btn_frame, text="Remove FG",
                   command=lambda: self.remove_fg("non_decision_makers")).pack(fill="x", pady=2)
        self.ndm_mute_btn = ttk.Button(ndm_btn_frame, text="Mute",
                                       command=lambda: self.toggle_mute_fg("non_decision_makers"))
        self.ndm_mute_btn.pack(fill="x", pady=2)

        # Track which list the FG editor is currently bound to
        self.active_fg_category = None

        # FG Editor for Decision Makers
        self.editor_frame = ttk.LabelFrame(self.project_inner, text="FG Editor")
        # Note: not packed here; on_fg_select shows/hides editors based on category.

        self.prop_vars = {}
        # (label, key, type, lo, hi)
        # lo/hi define the allowed range; rows show "[lo, hi]" label and a
        # slider next to the Entry. For type=="range" the range applies to
        # both Min and Max sub-fields.
        main_props = [
            ("Max Energy Reserve (ME_X MJ/ton)", "max_energy_reserve", "entry", 0.0, 10000.0),
            ("Energy Content (MJ/ton)", "energy_content", "entry", 0.0, 10000.0),
            ("Resting Metabolism (MJ/ton)", "resting_metabolism", "entry", 0.0, 1000.0),
            ("Maintenance Level (u_X, fraction)", "maintenance_level", "entry", 0.0, 1.0),
            ("Max Growth (MG_X, fraction/tick)", "growth_rate", "entry", 0.0, 1.0),
            ("Starve Rate (catabolism, fraction/tick)", "starve_rate", "entry", 0.0, 1.0),
            ("Visibility Floor (min visible fraction when hiding)", "visibility_floor", "entry", 0.0, 1.0),
            ("Natural Mortality (fraction/tick)", "natural_mortality", "entry", 0.0, 1.0),
            ("Max Intake Rate (ton prey / ton consumer / tick)", "max_intake_rate", "entry", 0.0, 1.0),
            ("Movement Speed (cells/tick)", "movement_speed", "entry", 0.0, 1.0),
            ("Indivisible Weight (kg)", "min_split_biomass", "entry", 0.0, 1000.0),
            ("Initial Total Biomass Range (ton)", "initial_biomass_range", "range", 0.0, 100000.0),
        ]

        # Add a thin separator under every variable/input row so the eye
        # can easily follow which Entry belongs to which label. Rows are
        # consumed sequentially via ``grow`` so multi-row entries (biomass
        # range, action costs) can take more than one row without a
        # separator between their sub-rows.
        self.editor_frame.columnconfigure(1, weight=1)
        grow = 0
        for (label, key, type, lo, hi) in main_props:
            ttk.Label(self.editor_frame, text=label).grid(row=grow, column=0, sticky="w", padx=5, pady=2)
            if type == "entry":
                var = self._build_ranged_entry(self.editor_frame, grow, lo, hi)
                self.prop_vars[key] = var
                grow += 1
            elif type == "range":
                # Min on its own row, Max on the next; sub-row labels go
                # into column 0 so the range-label/Entry/slider columns
                # line up with every other property row above.
                ttk.Label(self.editor_frame, text="  Min:").grid(
                    row=grow + 1, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(self.editor_frame, text="  Max:").grid(
                    row=grow + 2, column=0, sticky="w", padx=5, pady=2)
                min_var = self._build_ranged_entry(
                    self.editor_frame, grow + 1, lo, hi, is_int=True)
                max_var = self._build_ranged_entry(
                    self.editor_frame, grow + 2, lo, hi, is_int=True)
                self._link_minmax(min_var, max_var)
                self.prop_vars["initial_biomass_min"] = min_var
                self.prop_vars["initial_biomass_max"] = max_var
                grow += 3
            ttk.Separator(self.editor_frame, orient="horizontal").grid(
                row=grow, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 2))
            grow += 1

        # Action Costs: header on its own row, Eat and Move each on their
        # own row matching the standard ranged-entry column layout. No
        # separators between the three sub-rows.
        ttk.Label(self.editor_frame, text="Action Costs").grid(row=grow, column=0, sticky="w", padx=5, pady=2)
        action_costs = [
            ("  Eat:", "feeding_cost", 1.0, 10.0),
            ("  Move:", "movement_cost", 1.0, 10.0),
        ]
        for j, (label, key, lo, hi) in enumerate(action_costs):
            ttk.Label(self.editor_frame, text=label).grid(
                row=grow + 1 + j, column=0, sticky="w", padx=5, pady=2)
            var = self._build_ranged_entry(self.editor_frame, grow + 1 + j, lo, hi)
            self.prop_vars[key] = var
        grow += 1 + len(action_costs)

        ttk.Separator(self.editor_frame, orient="horizontal").grid(
            row=grow, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 2))
        grow += 1
        ttk.Button(self.editor_frame, text="Apply Changes", command=self.apply_fg_changes).grid(
            row=grow, column=0, columnspan=2, pady=5)
        grow += 1

        # Spawn Strategy editor (DM) — nested LabelFrame inside FG Editor
        self.spawn_frame = ttk.LabelFrame(self.editor_frame, text="Spawn Strategy")
        self.spawn_frame.grid(row=grow, column=0, columnspan=2,
                              sticky="ew", padx=5, pady=(8, 5))
        self.spawn_vars = {}
        self.spawn_vars["_owner_kind"] = "dm_fg"
        self._build_spawn_editor(self.spawn_frame, self.spawn_vars)

        # FG Editor for Non Decision Makers
        self.ndm_editor_frame = ttk.LabelFrame(self.project_inner, text="FG Editor")
        # Note: not packed here; on_fg_select shows/hides editors based on category.

        self.ndm_prop_vars = {}
        # (label, key, type, lo, hi)
        ndm_props = [
            ("Max Growth (fraction/tick)", "growth_rate", "entry", 0.0, 1.0),
            ("Max Carrying Capacity (ton/cell)", "max_carrying_capacity", "entry", 0.0, 100000.0),
            ("Energy Content (MJ/ton)", "energy_content", "entry", 0.0, 10000.0),
            ("Seed Rate (fraction of cc/tick)", "seed_rate", "entry", 0.0, 1.0),
            ("Seasonal Amplitude (fraction of growth_rate, 0=off)", "seasonal_amplitude", "entry", 0.0, 10.0),
            ("Seasonal Period (ticks)", "seasonal_period", "entry", 0.0, 10000.0),
            ("Initial Total Biomass Range (ton)", "initial_biomass_range", "range", 0.0, 100000.0),
        ]
        self.ndm_editor_frame.columnconfigure(1, weight=1)
        grow = 0
        for (label, key, type, lo, hi) in ndm_props:
            ttk.Label(self.ndm_editor_frame, text=label).grid(row=grow, column=0, sticky="w", padx=5, pady=2)
            if type == "range":
                ttk.Label(self.ndm_editor_frame, text="  Min:").grid(
                    row=grow + 1, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(self.ndm_editor_frame, text="  Max:").grid(
                    row=grow + 2, column=0, sticky="w", padx=5, pady=2)
                min_var = self._build_ranged_entry(
                    self.ndm_editor_frame, grow + 1, lo, hi, is_int=True)
                max_var = self._build_ranged_entry(
                    self.ndm_editor_frame, grow + 2, lo, hi, is_int=True)
                self._link_minmax(min_var, max_var)
                self.ndm_prop_vars["initial_biomass_min"] = min_var
                self.ndm_prop_vars["initial_biomass_max"] = max_var
                grow += 3
            else:
                var = self._build_ranged_entry(self.ndm_editor_frame, grow, lo, hi)
                self.ndm_prop_vars[key] = var
                grow += 1
            ttk.Separator(self.ndm_editor_frame, orient="horizontal").grid(
                row=grow, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 2))
            grow += 1

        ttk.Button(self.ndm_editor_frame, text="Apply Changes", command=self.apply_fg_changes).grid(
            row=grow, column=0, columnspan=2, pady=5
        )
        grow += 1

        # Spawn Strategy editor (NDM) — nested LabelFrame inside FG Editor
        self.ndm_spawn_frame = ttk.LabelFrame(self.ndm_editor_frame, text="Spawn Strategy")
        self.ndm_spawn_frame.grid(row=grow, column=0, columnspan=2,
                                  sticky="ew", padx=5, pady=(8, 5))
        self.ndm_spawn_vars = {}
        self.ndm_spawn_vars["_owner_kind"] = "ndm_fg"
        self._build_spawn_editor(self.ndm_spawn_frame, self.ndm_spawn_vars)

    # ------------------------------------------------------------------
    # Spawn-strategy editor (per-FG)
    # ------------------------------------------------------------------
    SPAWN_MODES = ("uniform", "perlin", "colony", "env_driven")
    # Display labels shown in the dropdown; internal keys remain lowercase.
    SPAWN_MODE_LABELS = {
        "uniform": "Uniform",
        "perlin": "Perlin",
        "colony": "Colony",
        "env_driven": "Env-driven",
    }
    SPAWN_MODE_KEYS = {v: k for k, v in {
        "uniform": "Uniform",
        "perlin": "Perlin",
        "colony": "Colony",
        "env_driven": "Env-driven",
    }.items()}
    SPAWN_MODE_HELP = {
        "uniform": (
            "Uniform\n"
            "\n"
            "Spreads biomass evenly across all allowed cells "
            "(Dirichlet-style), with no spatial correlation. This is "
            "the default behaviour and does not break any previous "
            "contracts. The per-cell floor (10·min_split) is still "
            "applied by the allocator.\n"
            "\n"
            "Parameters: none."
        ),
        "perlin": (
            "Perlin / fBm\n"
            "\n"
            "Creates coherent fields by summing low-pass filtered "
            "noise layers (fractal Brownian motion). Suitable for "
            "plankton, benthic communities and fish schools where "
            "the biology forms patches rather than an even mat.\n"
            "\n"
            "Scale: wavelength in cells — controls patch size. "
            "Larger = fewer, bigger patches.\n"
            "Octaves: number of frequency layers (1-6). More = finer "
            "structure on top of the base patches.\n"
            "Persistence: amplitude falloff per octave (0-1). Lower "
            "= cleaner low frequency.\n"
            "Lacunarity: frequency multiplier per octave (~2.0).\n"
            "Threshold: cell values below this are clamped to zero "
            "before normalisation (for sharper patch edges)."
        ),
        "colony": (
            "Colony\n"
            "\n"
            "Places N colony centres and smears the biomass with a "
            "Gaussian kernel around each centre. Intended for apex "
            "predators (seals, porpoises, seabirds) which have small "
            "populations in local aggregates, and addresses the "
            "documented 'frozen apex predator' effect where "
            "scattered spawning produces sub-threshold one-hot "
            "lock-in.\n"
            "\n"
            "N_colonies: number of colony centres.\n"
            "Sigma_cells: Gaussian width in cells around each centre.\n"
            "Anchor: free / coast / open_water. Reserved parameter "
            "— will be used once the official depth map is wired in; "
            "for now centres are chosen uniformly regardless of "
            "anchor.\n"
            "\n"
            "Amplitude_mode:\n"
            "  uniform (default, legacy): every centre contributes "
            "amplitude 1.0, colonies combine by SUM, and the field is "
            "min-max stretched so the strongest cell hits vmax. Best "
            "for biomass (sum-preserving).\n"
            "  jitter: every centre gets an INDEPENDENT amplitude "
            "Uniform(amplitude_min, amplitude_max) · vmax, colonies "
            "combine by MAX (not sum), and the field is NOT stretched "
            "— isolated centres land at exactly amp·vmax. Best for "
            "impacts that represent physical sources of different "
            "strength (e.g. noise sources at different dB levels) so "
            "the policy is trained against a heterogeneous mix of "
            "source intensities.\n"
            "Amplitude_min / amplitude_max: relative bounds in [0, 1] "
            "for the per-centre amplitude in jitter mode. Each colony "
            "centre value lies in [amplitude_min·vmax, amplitude_max·"
            "vmax]."
        ),
        "env_driven": (
            "Env-driven\n"
            "\n"
            "The per-cell weight is a linear combination of "
            "reference fields (e.g. another FG's biomass, future "
            "depth/light/nutrients) plus an optional Perlin overlay. "
            "Useful when a species' distribution should follow "
            "already computed layers — e.g. zooplankton following "
            "phytoplankton.\n"
            "\n"
            "Floor: minimum weight per cell before normalisation.\n"
            "Noise_amp: amplitude of the Perlin overlay (0 = none).\n"
            "Noise_scale: wavelength of the overlay in cells.\n"
            "\n"
            "Refs: each row references another FG's freshly spawned "
            "biomass field. Name = FG id (e.g. 'phytoplankton'), "
            "Weight = signed multiplier (negative inverts), Transform "
            "= linear/exp/invert/gauss_smooth. Use this to make "
            "zooplankton follow phytoplankton. Refs to 'depth' are "
            "filtered out until the depth map has been activated."
        ),
    }
    SPAWN_PARAM_SCHEMA = {
        "uniform": [],
        "perlin": [
            ("scale", "Scale (cells)", "float", 12.0),
            ("octaves", "Octaves", "int", 4),
            ("persistence", "Persistence", "float", 0.5),
            ("lacunarity", "Lacunarity", "float", 2.0),
            ("threshold", "Threshold", "float", 0.0),
        ],
        "colony": [
            ("n_colonies", "N colonies", "int", 3),
            ("sigma_cells", "Sigma (cells)", "float", 2.5),
            ("anchor", "Anchor", "choice:free,coast,open_water", "free"),
            ("amplitude_mode", "Amplitude mode",
             "choice:uniform,jitter", "uniform"),
            ("amplitude_min", "Amplitude min (rel.)", "float", 0.0),
            ("amplitude_max", "Amplitude max (rel.)", "float", 1.0),
        ],
        "env_driven": [
            ("floor", "Floor", "float", 0.0),
            ("noise_amp", "Noise amplitude", "float", 0.0),
            ("noise_scale", "Noise scale (cells)", "float", 4.0),
        ],
    }
    # Transform options available for env_driven refs (mirrors
    # lib/spawn/strategies.py: weights_env_driven).
    SPAWN_REF_TRANSFORMS = ("linear", "exp", "invert", "gauss_smooth")

    def _build_spawn_editor(self, parent, vars_store):
        """Build the per-FG spawn strategy editor inside ``parent``.

        ``parent`` is expected to be a dedicated container (e.g. a LabelFrame)
        that this method fills via ``pack``. Layout:
            - top row: mode dropdown
            - content row: parameters (left) | preview (middle) | Info (right)
        ``vars_store`` is populated with::
            'mode_var', 'param_vars' (dict key->tk.StringVar/BooleanVar),
            'param_frame', 'preview_canvas', '_preview_image' (kept alive),
            '_preview_after_id' (debounce token).
        """
        outer = ttk.Frame(parent)
        outer.pack(side="top", fill="x", anchor="w", padx=8, pady=6)

        # Mode row
        mode_row = ttk.Frame(outer)
        mode_row.pack(side="top", fill="x", anchor="w")
        ttk.Label(mode_row, text="Mode:").pack(side="left", padx=(0, 4))
        mode_var = tk.StringVar(value=self.SPAWN_MODE_LABELS["uniform"])
        mode_cb = ttk.Combobox(mode_row, textvariable=mode_var,
                               values=[self.SPAWN_MODE_LABELS[m] for m in self.SPAWN_MODES],
                               state="readonly", width=12)
        mode_cb.pack(side="left", padx=(0, 8))

        # Templates row: dropdown filtered by current mode + Save / Delete.
        # Templates are stored per (FG or impact) and per spawn mode in the
        # project file under ``spawn_templates: {<mode>: {<name>: {...}}}``;
        # see _on_save_spawn_template / _on_delete_spawn_template.
        tpl_row = ttk.Frame(outer)
        tpl_row.pack(side="top", fill="x", anchor="w", pady=(4, 0))
        ttk.Label(tpl_row, text="Templates:").pack(side="left", padx=(0, 4))
        template_var = tk.StringVar(value="")
        template_cb = ttk.Combobox(tpl_row, textvariable=template_var,
                                   values=[], state="readonly", width=20)
        template_cb.pack(side="left", padx=(0, 8))
        save_btn = ttk.Button(
            tpl_row, text="Save",
            command=lambda vs=vars_store: self._on_save_spawn_template(vs))
        save_btn.pack(side="left", padx=(0, 4))
        delete_btn = ttk.Button(
            tpl_row, text="Delete",
            command=lambda vs=vars_store: self._on_delete_spawn_template(vs))
        delete_btn.pack(side="left")

        vars_store["template_var"] = template_var
        vars_store["template_cb"] = template_cb
        vars_store["template_save_btn"] = save_btn
        vars_store["template_delete_btn"] = delete_btn

        def _on_template_pick(*_a):
            name = template_var.get()
            if not name:
                return
            self._on_apply_spawn_template(vars_store, name)

        template_cb.bind("<<ComboboxSelected>>", _on_template_pick)

        # Content row: params on left, preview on right
        content = ttk.Frame(outer)
        content.pack(side="top", fill="x", anchor="w", pady=(4, 0))

        param_frame = ttk.Frame(content)
        param_frame.pack(side="left", anchor="nw", padx=(0, 12))

        preview_frame = ttk.Frame(content)
        preview_frame.pack(side="left", anchor="nw")
        ttk.Label(preview_frame, text="Preview", font=("TkDefaultFont", 8)).pack(anchor="w")
        preview_canvas = tk.Canvas(preview_frame, width=120, height=120,
                                   bg="#202020", highlightthickness=1,
                                   highlightbackground="#888888")
        preview_canvas.pack(anchor="w")

        # Help panel to the right of the preview, inside the FG editor frame.
        help_frame = ttk.LabelFrame(content, text="Info")
        help_frame.pack(side="left", anchor="nw", padx=(12, 0), fill="y")
        help_label = ttk.Label(help_frame, text="", justify="left",
                               wraplength=260, anchor="nw",
                               font=("TkDefaultFont", 8))
        help_label.pack(side="top", anchor="nw", padx=6, pady=4)

        vars_store["mode_var"] = mode_var
        vars_store["param_vars"] = {}
        vars_store["param_frame"] = param_frame
        vars_store["preview_canvas"] = preview_canvas
        vars_store["help_label"] = help_label
        vars_store["_preview_image"] = None
        vars_store["_preview_after_id"] = None
        vars_store["_param_widgets"] = []

        def _on_mode_change(*_a):
            self._rebuild_spawn_params(vars_store)
            self._update_spawn_help(vars_store)
            self._schedule_spawn_preview(vars_store)
            # Templates are per-mode: refresh the dropdown so only
            # entries saved under the current mode are listed.
            self._refresh_spawn_template_list(vars_store)

        mode_var.trace_add("write", _on_mode_change)

        # Initial build (uniform → empty param frame, but still schedules preview)
        self._rebuild_spawn_params(vars_store)
        self._update_spawn_help(vars_store)
        self._schedule_spawn_preview(vars_store)
        self._refresh_spawn_template_list(vars_store)

    def _update_spawn_help(self, vars_store):
        """Update the help panel text for the currently selected spawn mode."""
        lbl = vars_store.get("help_label")
        if lbl is None:
            return
        mode = self.SPAWN_MODE_KEYS.get(vars_store["mode_var"].get(), vars_store["mode_var"].get())
        text = self.SPAWN_MODE_HELP.get(mode, "")
        try:
            lbl.configure(text=text)
        except Exception:
            pass

    def _rebuild_spawn_params(self, vars_store):
        """Rebuild the parameter sub-frame to match the current mode."""
        frame = vars_store["param_frame"]
        for w in vars_store.get("_param_widgets", []):
            try:
                w.destroy()
            except Exception:
                pass
        vars_store["_param_widgets"] = []
        vars_store["param_vars"] = {}
        # Reset refs-editor state; only env_driven repopulates it below.
        vars_store["refs_rows"] = []
        vars_store["refs_frame"] = None

        mode = self.SPAWN_MODE_KEYS.get(vars_store["mode_var"].get(), vars_store["mode_var"].get())
        schema = self.SPAWN_PARAM_SCHEMA.get(mode, [])

        if not schema and mode != "env_driven":
            placeholder = ttk.Label(frame, text="(no parameters)",
                                    foreground="#888888")
            placeholder.grid(row=0, column=0, sticky="w")
            vars_store["_param_widgets"].append(placeholder)
            return

        next_row = 0
        for i, (key, label, ptype, default) in enumerate(schema):
            lbl = ttk.Label(frame, text=label + ":")
            lbl.grid(row=i, column=0, sticky="w", padx=(0, 4), pady=1)
            vars_store["_param_widgets"].append(lbl)

            if ptype.startswith("choice:"):
                choices = ptype.split(":", 1)[1].split(",")
                var = tk.StringVar(value=str(default))
                w = ttk.Combobox(frame, textvariable=var, values=choices,
                                 state="readonly", width=12)
            else:
                var = tk.StringVar(value=str(default))
                w = ttk.Entry(frame, textvariable=var, width=10)
            w.grid(row=i, column=1, sticky="w", pady=1)
            vars_store["_param_widgets"].append(w)
            vars_store["param_vars"][key] = (var, ptype)
            # Live-update preview on edit
            var.trace_add("write", lambda *_a, vs=vars_store: self._schedule_spawn_preview(vs))
            next_row = i + 1

        # Env-driven: build refs editor under the scalar params.
        if mode == "env_driven":
            self._build_refs_editor(frame, vars_store, start_row=next_row)

    def _build_refs_editor(self, parent, vars_store, start_row):
        """Build the env_driven refs editor (list of {name, weight, transform}).

        Each row exposes:
          - Name: dropdown of currently known FG ids (project + library)
          - Weight: signed float
          - Transform: linear / exp / invert / gauss_smooth
          - Remove button (X)
        Plus an "Add ref" button below the rows.
        """
        # Header
        header = ttk.Label(parent, text="Refs (env layers):",
                          font=("TkDefaultFont", 9, "bold"))
        header.grid(row=start_row, column=0, columnspan=4,
                    sticky="w", pady=(6, 2))
        vars_store["_param_widgets"].append(header)

        refs_frame = ttk.Frame(parent)
        refs_frame.grid(row=start_row + 1, column=0, columnspan=4,
                        sticky="w")
        vars_store["_param_widgets"].append(refs_frame)
        vars_store["refs_frame"] = refs_frame

        # Sub-header row inside refs_frame
        ttk.Label(refs_frame, text="Name", font=("TkDefaultFont", 8)).grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Label(refs_frame, text="Weight", font=("TkDefaultFont", 8)).grid(
            row=0, column=1, sticky="w", padx=(0, 4))
        ttk.Label(refs_frame, text="Transform", font=("TkDefaultFont", 8)).grid(
            row=0, column=2, sticky="w", padx=(0, 4))

        add_btn = ttk.Button(parent, text="+ Add ref",
                             command=lambda vs=vars_store: self._add_ref_row(vs))
        add_btn.grid(row=start_row + 2, column=0, columnspan=2,
                     sticky="w", pady=(4, 2))
        vars_store["_param_widgets"].append(add_btn)

    def _known_fg_ids(self):
        """Return the list of FG ids known to the project, then library."""
        ids = []
        for fg in (self.project_data.get("decision_makers", []) or []):
            gid = fg.get("group_id") if isinstance(fg, dict) else None
            if gid and gid not in ids:
                ids.append(gid)
        for fg in (self.project_data.get("non_decision_makers", []) or []):
            gid = fg.get("group_id") if isinstance(fg, dict) else None
            if gid and gid not in ids:
                ids.append(gid)
        # Add any library entries not yet in the project (helps the user
        # pre-configure refs before the dependency FG is added).
        for sid in (self.global_library.get("species_definitions", {}) or {}):
            if sid not in ids:
                ids.append(sid)
        return ids

    def _add_ref_row(self, vars_store, name="", weight=1.0, transform="linear"):
        """Append a new ref row to the env_driven refs editor."""
        refs_frame = vars_store.get("refs_frame")
        if refs_frame is None or not refs_frame.winfo_exists():
            return
        rows = vars_store.setdefault("refs_rows", [])
        row_idx = len(rows) + 1  # row 0 is the header
        name_var = tk.StringVar(value=str(name))
        weight_var = tk.StringVar(value=str(weight))
        transform_var = tk.StringVar(value=str(transform))
        name_cb = ttk.Combobox(refs_frame, textvariable=name_var,
                                values=self._known_fg_ids(), width=18)
        name_cb.grid(row=row_idx, column=0, sticky="w", padx=(0, 4), pady=1)
        weight_ent = ttk.Entry(refs_frame, textvariable=weight_var, width=8)
        weight_ent.grid(row=row_idx, column=1, sticky="w", padx=(0, 4), pady=1)
        transform_cb = ttk.Combobox(refs_frame, textvariable=transform_var,
                                    values=list(self.SPAWN_REF_TRANSFORMS),
                                    state="readonly", width=12)
        transform_cb.grid(row=row_idx, column=2, sticky="w", padx=(0, 4), pady=1)
        row_data = {
            "name_var": name_var, "weight_var": weight_var,
            "transform_var": transform_var,
            "widgets": [name_cb, weight_ent, transform_cb],
        }
        # Remove button
        def _remove(rd=row_data, vs=vars_store):
            try:
                for w in rd["widgets"]:
                    w.destroy()
                if rd.get("remove_btn") is not None:
                    rd["remove_btn"].destroy()
            except Exception:
                pass
            try:
                vs["refs_rows"].remove(rd)
            except ValueError:
                pass
            self._schedule_spawn_preview(vs)
        rm_btn = ttk.Button(refs_frame, text="X", width=2, command=_remove)
        rm_btn.grid(row=row_idx, column=3, sticky="w", pady=1)
        row_data["remove_btn"] = rm_btn
        rows.append(row_data)
        # Live preview updates
        for v in (name_var, weight_var, transform_var):
            v.trace_add("write", lambda *_a, vs=vars_store: self._schedule_spawn_preview(vs))
        self._schedule_spawn_preview(vars_store)

    def _schedule_spawn_preview(self, vars_store, delay_ms=250):
        """Debounce preview re-rendering so we don't recompute on every keystroke."""
        canvas = vars_store.get("preview_canvas")
        if canvas is None or not canvas.winfo_exists():
            return
        prev_id = vars_store.get("_preview_after_id")
        if prev_id is not None:
            try:
                canvas.after_cancel(prev_id)
            except Exception:
                pass
        vars_store["_preview_after_id"] = canvas.after(
            delay_ms, lambda: self._render_spawn_preview(vars_store))

    def _collect_spawn_dict(self, vars_store, errors=None):
        """Collect current editor state into a YAML-shaped dict.

        If ``errors`` is a list it is populated with human-readable
        descriptions of any unparseable inputs (spawn params or ref
        weights). Callers that pass a list should treat a non-empty
        result as a hard validation failure (do not silently save).
        When ``errors`` is None we preserve the legacy lenient behaviour
        for the live preview path so typing intermediate values like
        '0.' doesn't blank the canvas.
        """
        strict = errors is not None
        mode = self.SPAWN_MODE_KEYS.get(vars_store["mode_var"].get(), vars_store["mode_var"].get())
        out = {"mode": mode}
        for key, (var, ptype) in vars_store.get("param_vars", {}).items():
            raw = var.get()
            if ptype == "int":
                try:
                    out[key] = int(float(raw))
                except (TypeError, ValueError):
                    if strict and str(raw).strip() != "":
                        errors.append(
                            f"Spawn parameter '{key}' is not a valid integer: {raw!r}"
                        )
                    continue
            elif ptype == "float":
                try:
                    out[key] = float(raw)
                except (TypeError, ValueError):
                    if strict and str(raw).strip() != "":
                        errors.append(
                            f"Spawn parameter '{key}' is not a valid number: {raw!r}"
                        )
                    continue
            else:
                # choice or unknown → store as string
                if raw != "":
                    out[key] = raw
        # env_driven: serialise the refs list. Rows with empty name are
        # silently dropped; invalid weights are reported in strict mode
        # (Apply) and default to 1.0 only in lenient mode (live preview).
        if mode == "env_driven":
            refs_out = []
            for rd in vars_store.get("refs_rows", []) or []:
                name = rd["name_var"].get().strip()
                if not name:
                    continue
                raw_w = rd["weight_var"].get()
                try:
                    weight = float(raw_w)
                except (TypeError, ValueError):
                    if strict:
                        errors.append(
                            f"Ref '{name}' has an invalid weight: {raw_w!r}"
                        )
                        continue
                    weight = 1.0
                transform = rd["transform_var"].get().strip() or "linear"
                refs_out.append({"name": name, "weight": weight,
                                 "transform": transform})
            if refs_out:
                out["refs"] = refs_out
        return out

    def _get_reference_grid(self):
        """Return (W, H) reference grid from project_metadata, defaulting to 60x60."""
        try:
            w = int(self.ref_grid_w_var.get())
        except (AttributeError, TypeError, ValueError):
            w = 60
        try:
            h = int(self.ref_grid_h_var.get())
        except (AttributeError, TypeError, ValueError):
            h = 60
        if w < 3:
            w = 60
        if h < 3:
            h = 60
        return w, h

    def _render_spawn_preview(self, vars_store):
        """Compute weights via lib.spawn.make_weights and draw on the preview canvas."""
        canvas = vars_store.get("preview_canvas")
        if canvas is None or not canvas.winfo_exists():
            return
        try:
            import os, sys
            # Ensure project root is on sys.path so `lib.spawn` is importable
            # regardless of cwd (fgconfig.py may run from fgconfig/ or root).
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.dirname(_here)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            import numpy as np
            from lib.spawn import make_weights
            from lib.spawn.strategies import StrategySpec
        except Exception as exc:
            canvas.delete("all")
            canvas.create_text(60, 60, text=f"preview error\n{type(exc).__name__}: {str(exc)[:30]}",
                               fill="#cccccc", font=("TkDefaultFont", 7),
                               justify="center")
            return

        ref_w, ref_h = self._get_reference_grid()
        spec_dict = self._collect_spawn_dict(vars_store)
        try:
            spec = StrategySpec.from_dict(spec_dict)
            ctx = {"biomass_scale": 1.0}
            # For env_driven previews: synthesize a smooth dummy field for
            # every referenced FG so the user can visualise how the chosen
            # refs/weights/transforms shape the resulting distribution. Each
            # ref name gets a distinct Perlin-style pattern (seeded by the
            # hash of the name) so different refs are clearly visible.
            if spec_dict.get("mode") == "env_driven" and spec_dict.get("refs"):
                from lib.spawn.strategies import _gaussian_random_field
                env_fields_preview = {}
                for r in spec_dict.get("refs", []):
                    nm = r.get("name") if isinstance(r, dict) else None
                    if not nm:
                        continue
                    seed = (abs(hash(nm)) & 0x7FFFFFFF) or 1
                    env_fields_preview[nm] = _gaussian_random_field(
                        ref_h, ref_w, scale=12.0, octaves=4,
                        persistence=0.5, lacunarity=2.0, seed=seed)
                ctx["env_fields"] = env_fields_preview
            weights = make_weights(spec, (ref_h, ref_w), project_seed=0, context=ctx)
        except Exception as exc:
            canvas.delete("all")
            canvas.create_text(60, 60,
                               text=f"{type(exc).__name__}\n{str(exc)[:40]}",
                               fill="#ffaaaa", font=("TkDefaultFont", 7),
                               justify="center")
            return

        # Normalize to [0,1] for grayscale rendering
        w_max = float(weights.max()) if weights.size else 0.0
        if w_max <= 0:
            norm = np.zeros_like(weights, dtype=float)
        else:
            norm = weights / w_max

        # Build a 120x120 PhotoImage by nearest-neighbour upscaling
        out_size = 120
        H, W = norm.shape
        # Build PPM (P6) header + bytes — fastest reliable Tk image format
        scale_x = W / out_size
        scale_y = H / out_size
        # Vectorized resample
        xs = (np.arange(out_size) * scale_x).astype(int).clip(0, W - 1)
        ys = (np.arange(out_size) * scale_y).astype(int).clip(0, H - 1)
        resampled = norm[ys[:, None], xs[None, :]]
        # Apply a simple viridis-like colormap (dark blue → green → yellow)
        r = np.clip(resampled * 1.4 - 0.4, 0, 1)
        g = np.clip(resampled * 1.2, 0, 1)
        b = np.clip(0.6 - resampled * 0.6 + 0.2, 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        header = f"P6 {out_size} {out_size} 255 ".encode("ascii")
        ppm = header + rgb.tobytes()
        try:
            img = tk.PhotoImage(data=ppm, format="PPM")
        except tk.TclError:
            # Fallback: tiny fallback message if Tk lacks PPM support
            canvas.delete("all")
            canvas.create_text(60, 60, text="PPM unsupported",
                               fill="#cccccc", font=("TkDefaultFont", 7))
            return
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=img)
        vars_store["_preview_image"] = img  # keep ref alive

    def _populate_spawn_editor(self, vars_store, spawn_cfg):
        """Populate the spawn editor from a YAML-shaped spawn dict."""
        if not isinstance(spawn_cfg, dict):
            spawn_cfg = {}
        mode = spawn_cfg.get("mode", "uniform")
        if mode not in self.SPAWN_MODES:
            mode = "uniform"
        # Setting mode_var triggers _rebuild_spawn_params via trace
        vars_store["mode_var"].set(self.SPAWN_MODE_LABELS[mode])
        # After rebuild, fill in any matching params
        for key, (var, ptype) in vars_store.get("param_vars", {}).items():
            if key in spawn_cfg:
                var.set(str(spawn_cfg[key]))
        # env_driven: rebuild refs rows from YAML list
        if mode == "env_driven":
            refs = spawn_cfg.get("refs") or []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                name = ref.get("name", "")
                weight = ref.get("weight", 1.0)
                transform = ref.get("transform", "linear")
                self._add_ref_row(vars_store, name=name, weight=weight,
                                  transform=transform)
        self._schedule_spawn_preview(vars_store, delay_ms=50)
        # Refresh the Templates dropdown for the (newly selected) owner.
        self._refresh_spawn_template_list(vars_store)

    # ------------------------------------------------------------------
    # Spawn templates (global, per-mode)
    # ------------------------------------------------------------------
    # Templates are stored globally in the project file at the top level
    # under ``spawn_templates``, keyed by spawn mode
    # (uniform / perlin / colony / env_driven), then by user-chosen name.
    # The value is the spawn parameter dict (without the ``mode`` key,
    # which is implied by the parent key). Example::
    #
    #     spawn_templates:
    #       perlin:
    #         coastal: {scale: 12.0, octaves: 4, persistence: 0.5, ...}
    #       colony:
    #         five_blobs: {n_colonies: 5, sigma_cells: 8.0, ...}
    #
    # Templates are shared across all FGs and impacts. Only templates
    # saved under the currently-selected Mode appear in the Templates
    # dropdown.

    def _get_spawn_templates_dict(self, create=False):
        """Return the global ``project_data['spawn_templates']`` dict, or {}.

        When ``create`` is True, the key is added to ``project_data``
        if missing.
        """
        if not isinstance(getattr(self, "project_data", None), dict):
            return {}
        tpl = self.project_data.get("spawn_templates")
        if not isinstance(tpl, dict):
            if create:
                tpl = {}
                self.project_data["spawn_templates"] = tpl
            else:
                return {}
        return tpl

    def _refresh_spawn_template_list(self, vars_store):
        """Repopulate the Templates dropdown for the current owner+mode."""
        cb = vars_store.get("template_cb")
        var = vars_store.get("template_var")
        if cb is None or var is None:
            return
        try:
            if not cb.winfo_exists():
                return
        except Exception:
            return
        mode = self.SPAWN_MODE_KEYS.get(
            vars_store["mode_var"].get(), vars_store["mode_var"].get())
        names = []
        tpl_root = self._get_spawn_templates_dict()
        mode_tpls = tpl_root.get(mode)
        if isinstance(mode_tpls, dict):
            names = sorted(mode_tpls.keys())
        try:
            cb.configure(values=names)
        except Exception:
            pass
        # Clear current selection — picking a template is an explicit
        # user action; we never auto-apply on refresh.
        var.set("")

    def _on_save_spawn_template(self, vars_store):
        """Save the current GUI spawn settings as a named template."""
        mode = self.SPAWN_MODE_KEYS.get(
            vars_store["mode_var"].get(), vars_store["mode_var"].get())
        # Use strict mode so invalid GUI values surface as an error
        # instead of being silently dropped from the saved template.
        errors = []
        spec = self._collect_spawn_dict(vars_store, errors=errors)
        if errors:
            messagebox.showerror(
                "Invalid spawn settings",
                "Cannot save template — the spawn editor has invalid values:\n\n  - "
                + "\n  - ".join(errors))
            return
        # The mode is implied by the parent dict key; strip it from the
        # stored payload to keep YAML diffs clean.
        spec.pop("mode", None)

        # Ask for a name. Pre-fill with the current dropdown value if any.
        from tkinter import simpledialog
        current = vars_store["template_var"].get().strip()
        name = simpledialog.askstring(
            "Save template",
            f"Template name (mode={mode}):",
            initialvalue=current,
            parent=self.root if hasattr(self, "root") else None)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Empty name",
                                   "Template name cannot be empty.")
            return

        tpl_root = self._get_spawn_templates_dict(create=True)
        mode_tpls = tpl_root.get(mode)
        if not isinstance(mode_tpls, dict):
            mode_tpls = {}
            tpl_root[mode] = mode_tpls
        if name in mode_tpls:
            if not messagebox.askyesno(
                    "Overwrite template",
                    f"A template named '{name}' already exists for mode "
                    f"'{mode}'. Overwrite?"):
                return
        mode_tpls[name] = spec
        self._mark_dirty()
        self._refresh_spawn_template_list(vars_store)
        try:
            vars_store["template_var"].set(name)
        except Exception:
            pass

    def _on_delete_spawn_template(self, vars_store):
        """Delete the currently-selected global template."""
        name = vars_store["template_var"].get().strip()
        if not name:
            messagebox.showinfo(
                "No template selected",
                "Pick a template from the dropdown before deleting.")
            return
        mode = self.SPAWN_MODE_KEYS.get(
            vars_store["mode_var"].get(), vars_store["mode_var"].get())
        tpl_root = self._get_spawn_templates_dict()
        mode_tpls = tpl_root.get(mode) if isinstance(tpl_root, dict) else None
        if not isinstance(mode_tpls, dict) or name not in mode_tpls:
            return
        if not messagebox.askyesno(
                "Delete template",
                f"Delete template '{name}' (mode={mode})?"):
            return
        del mode_tpls[name]
        # Clean up empty containers so YAML stays tidy.
        if not mode_tpls:
            tpl_root.pop(mode, None)
        if isinstance(tpl_root, dict) and not tpl_root:
            self.project_data.pop("spawn_templates", None)
        self._mark_dirty()
        self._refresh_spawn_template_list(vars_store)

    def _on_apply_spawn_template(self, vars_store, name):
        """Apply the named template's values to the editor (GUI only).

        The user must still click ``Apply Changes`` to persist the
        values onto the FG / impact's own ``spawn`` block.
        """
        mode = self.SPAWN_MODE_KEYS.get(
            vars_store["mode_var"].get(), vars_store["mode_var"].get())
        tpl_root = self._get_spawn_templates_dict()
        mode_tpls = tpl_root.get(mode) if isinstance(tpl_root, dict) else None
        if not isinstance(mode_tpls, dict):
            return
        spec = mode_tpls.get(name)
        if not isinstance(spec, dict):
            return
        # Reconstruct a full spawn dict (mode is implied by the parent key).
        full = {"mode": mode}
        full.update(spec)
        # _populate_spawn_editor will set mode_var (no-op here, same mode)
        # and refresh params / preview / template list. To preserve the
        # current selection in the dropdown, restore template_var after.
        self._populate_spawn_editor(vars_store, full)
        try:
            vars_store["template_var"].set(name)
        except Exception:
            pass

    def _read_initial_biomass_range(self, fg_entry, fg_id):
        """Read (min, max) initial biomass for a project FG entry.

        The range is stored per-project on the FG entry as
        ``initial_biomass_min``/``initial_biomass_max``. The library no longer
        provides defaults.
        Returns (min_or_None, max_or_None) as ints (or None when unset/invalid).
        """
        def _coerce_int(v):
            if v is None or v == "":
                return None
            try:
                iv = int(round(float(v)))
                return iv if iv >= 0 else None
            except (TypeError, ValueError):
                return None

        if isinstance(fg_entry, dict):
            mn = _coerce_int(fg_entry.get("initial_biomass_min"))
            mx = _coerce_int(fg_entry.get("initial_biomass_max"))
            if mn is not None or mx is not None:
                return mn, mx
        return None, None

    def _build_biomass_range_row(self, parent, row, lo=0.0, hi=100000.0):
        """Place a Min/Max entry pair side-by-side for the 'Initial Total Biomass Range (ton)' row.

        Each sub-field has its own ``[lo, hi]`` range label and slider on the
        right. Returns (min_var, max_var). Live validation: non-negative
        integers within [lo, hi], with the cross-constraint min <= max.
        """
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        min_var = tk.StringVar()
        max_var = tk.StringVar()

        lo_i = int(lo)
        hi_i = int(hi)

        def _is_pos_int_in_range(s):
            if s == "":
                return True
            if not s.isdigit():
                return False
            try:
                v = int(s)
            except ValueError:
                return False
            return lo_i <= v <= hi_i

        def _validate_min(proposed):
            if not _is_pos_int_in_range(proposed):
                return False
            if proposed == "":
                return True
            cur_max = max_var.get()
            if cur_max.isdigit() and int(proposed) > int(cur_max):
                return False
            return True

        def _validate_max(proposed):
            if not _is_pos_int_in_range(proposed):
                return False
            if proposed == "":
                return True
            cur_min = min_var.get()
            if cur_min.isdigit() and int(proposed) < int(cur_min):
                return False
            return True

        vcmd_min = (parent.register(_validate_min), "%P")
        vcmd_max = (parent.register(_validate_max), "%P")

        ttk.Label(container, text="Min:").pack(side="left", padx=(0, 2))
        ttk.Label(container, text=f"[{lo_i}, {hi_i}]",
                  foreground="#666666", anchor="e", width=14
                  ).pack(side="left", padx=(0, 2))
        min_entry = ttk.Entry(container, textvariable=min_var, width=10,
                              validate="key", validatecommand=vcmd_min)
        min_entry.pack(side="left", padx=(0, 4))
        min_slider = ttk.Scale(container, from_=lo_i, to=hi_i,
                               orient="horizontal", length=120)
        min_slider.pack(side="left", padx=(0, 12))
        self._bind_slider_entry(min_var, min_slider, lo_i, hi_i, is_int=True)

        ttk.Label(container, text="Max:").pack(side="left", padx=(0, 2))
        ttk.Label(container, text=f"[{lo_i}, {hi_i}]",
                  foreground="#666666", anchor="e", width=14
                  ).pack(side="left", padx=(0, 2))
        max_entry = ttk.Entry(container, textvariable=max_var, width=10,
                              validate="key", validatecommand=vcmd_max)
        max_entry.pack(side="left", padx=(0, 4))
        max_slider = ttk.Scale(container, from_=lo_i, to=hi_i,
                               orient="horizontal", length=120)
        max_slider.pack(side="left")
        self._bind_slider_entry(max_var, max_slider, lo_i, hi_i, is_int=True)

        return min_var, max_var

    def _bind_slider_entry(self, var, slider, lo, hi, is_int=False):
        """Two-way binding between a tk.StringVar (Entry) and a ttk.Scale.

        Updates the slider position when the Entry changes (if the value
        parses and falls within [lo, hi]), and writes back to the Entry
        when the slider is dragged. Live in-range validation already
        happens on the Entry side; this method only mirrors values.
        """
        state = {"sync": False}

        def _entry_to_slider(*_a):
            if state["sync"]:
                return
            raw = var.get()
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return
            if v < lo or v > hi:
                return
            state["sync"] = True
            try:
                slider.set(v)
            finally:
                state["sync"] = False

        def _slider_to_entry(val):
            if state["sync"]:
                return
            try:
                v = float(val)
            except (TypeError, ValueError):
                return
            if is_int:
                v = int(round(v))
                txt = str(v)
            else:
                # Render with reasonable precision; %g trims trailing zeros.
                txt = f"{v:.4g}"
            state["sync"] = True
            try:
                var.set(txt)
            finally:
                state["sync"] = False

        var.trace_add("write", _entry_to_slider)
        slider.configure(command=_slider_to_entry)
        # Initial sync (if var already holds a value).
        _entry_to_slider()

    def _build_ranged_entry(self, parent, row, lo, hi, is_int=None):
        """Build a [lo, hi] range label + Entry + slider triple in column 1.

        Returns the tk.StringVar bound to the Entry (kept as StringVar so
        the existing apply_fg_changes / on_fg_select code paths continue
        to read/write it untouched). Live validation rejects typed values
        outside [lo, hi]; the slider is kept in sync both directions.
        """
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        container.columnconfigure(1, weight=1)
        return self._build_inline_ranged_entry(container, lo, hi,
                                               pack=False, width=12,
                                               is_int=is_int)

    def _link_minmax(self, min_var, max_var):
        """Enforce min <= max across two StringVars on every write.

        Uses ``trace_add('write', ...)`` to clamp the offending side once
        a constraint violation appears. This complements the per-field
        live validation (which only sees its own field's proposed value).
        """
        state = {"sync": False}

        def _as_f(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        def _on_min(*_a):
            if state["sync"]:
                return
            lo = _as_f(min_var.get())
            hi = _as_f(max_var.get())
            if lo is not None and hi is not None and lo > hi:
                state["sync"] = True
                try:
                    max_var.set(min_var.get())
                finally:
                    state["sync"] = False

        def _on_max(*_a):
            if state["sync"]:
                return
            lo = _as_f(min_var.get())
            hi = _as_f(max_var.get())
            if lo is not None and hi is not None and hi < lo:
                state["sync"] = True
                try:
                    min_var.set(max_var.get())
                finally:
                    state["sync"] = False

        min_var.trace_add("write", _on_min)
        max_var.trace_add("write", _on_max)

    def _build_inline_ranged_entry(self, container, lo, hi, *,
                                   pack=True, width=12, is_int=None):
        """Place a [lo, hi] label + Entry + slider into ``container``.

        ``pack=True`` uses .pack() (for the Action Costs row that itself
        uses pack), ``pack=False`` uses .grid() (for the multi-row FG
        editor where the outer container has columnconfigure).
        Returns the tk.StringVar bound to the Entry.
        """
        var = tk.StringVar()

        def _validate(proposed):
            if proposed in ("", ".", "-", "-.", "+", "+."):
                return True
            try:
                v = float(proposed)
            except ValueError:
                return False
            return lo <= v <= hi

        vcmd = (container.register(_validate), "%P")

        # Right-align the range label inside a fixed-width slot so that
        # the Entry columns line up across rows regardless of how wide the
        # "[lo, hi]" text is.
        lbl = ttk.Label(container, text=f"[{lo:g}, {hi:g}]",
                        foreground="#666666", anchor="e", width=16)
        ent = ttk.Entry(container, textvariable=var, width=width,
                        validate="key", validatecommand=vcmd)
        scl = ttk.Scale(container, from_=lo, to=hi,
                        orient="horizontal", length=140)

        if pack:
            lbl.pack(side="left", padx=(0, 4))
            ent.pack(side="left", padx=(0, 4))
            scl.pack(side="left", padx=(0, 10))
        else:
            lbl.grid(row=0, column=0, sticky="e", padx=(0, 4))
            ent.grid(row=0, column=1, sticky="ew", padx=(0, 4))
            scl.grid(row=0, column=2, sticky="ew", padx=(0, 4))
            container.columnconfigure(2, weight=2)

        # Use integer-rounding either when explicitly requested, or as a
        # fallback heuristic for wide integer ranges (e.g. action costs).
        if is_int is None:
            is_int_range = (lo == int(lo) and hi == int(hi) and (hi - lo) >= 1.0
                            and hi >= 10)
        else:
            is_int_range = bool(is_int)
        self._bind_slider_entry(var, scl, lo, hi, is_int=is_int_range)
        return var

    def _build_value_range_row(self, parent, row):
        """Build a Min/Max entry pair (side by side) for an impact value range.

        Accepts non-negative floats; live validation forbids typing a max
        smaller than the current min (and vice versa). Empty values are
        allowed during editing. Returns (min_var, max_var).
        """
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="w", padx=5, pady=2)

        min_var = tk.StringVar()
        max_var = tk.StringVar()

        def _is_nonneg_float(s):
            if s == "":
                return True
            try:
                return float(s) >= 0
            except ValueError:
                return False

        def _as_f(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        def _validate_min(proposed):
            if not _is_nonneg_float(proposed):
                return False
            if proposed == "":
                return True
            cur_max = _as_f(max_var.get())
            if cur_max is not None and float(proposed) > cur_max:
                return False
            return True

        def _validate_max(proposed):
            if not _is_nonneg_float(proposed):
                return False
            if proposed == "":
                return True
            cur_min = _as_f(min_var.get())
            if cur_min is not None and float(proposed) < cur_min:
                return False
            return True

        vcmd_min = (parent.register(_validate_min), "%P")
        vcmd_max = (parent.register(_validate_max), "%P")

        ttk.Label(container, text="Min:").pack(side="left", padx=(0, 2))
        min_entry = ttk.Entry(container, textvariable=min_var, width=10,
                              validate="key", validatecommand=vcmd_min)
        min_entry.pack(side="left", padx=(0, 8))
        ttk.Label(container, text="Max:").pack(side="left", padx=(0, 2))
        max_entry = ttk.Entry(container, textvariable=max_var, width=10,
                              validate="key", validatecommand=vcmd_max)
        max_entry.pack(side="left")
        return min_var, max_var

    def _sync_impact_spawn_visibility(self):
        """Show the Spawn Strategy frame iff the impact is observable.

        Non-observable impacts are zero-filled at training time, so the
        spawn-strategy parameters would have no effect — we hide the
        whole LabelFrame to make this clear.
        """
        if not hasattr(self, 'impact_spawn_frame'):
            return
        try:
            if bool(self.impact_observable_var.get()):
                self.impact_spawn_frame.grid()
            else:
                self.impact_spawn_frame.grid_remove()
        except tk.TclError:
            pass

    def _on_impact_observable_toggled(self):
        """Checkbox callback: toggle Spawn-Strategy visibility live."""
        self._sync_impact_spawn_visibility()

    def _set_impact_editor_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        # Recursively walk the editor frame and disable inputs/buttons.
        def _walk(w):
            for c in w.winfo_children():
                try:
                    c.configure(state=state)
                except tk.TclError:
                    pass
                _walk(c)
        _walk(self.impact_editor_frame)

    def _selected_impact_entry(self):
        """Return the impact_variables entry currently selected, or None."""
        try:
            sel = self.impact_listbox.curselection()
        except tk.TclError:
            return None
        if not sel:
            return None
        idx = sel[0]
        ivs = self.project_data.get('impact_variables', []) or []
        if idx >= len(ivs):
            return None
        return ivs[idx]

    def on_impact_select(self):
        entry = self._selected_impact_entry()
        if entry is None:
            self.impact_editor_label_var.set("(no impact selected)")
            self.impact_value_min_var.set("")
            self.impact_value_max_var.set("")
            self._set_impact_editor_enabled(False)
            return
        impact_id = entry.get('impact_id', '')
        display = self.global_library.get("impact_definitions", {}).get(
            impact_id, {}).get("display_name", impact_id)
        unit = self._impact_unit(impact_id)
        title = f"Impact Editor — {display} ({unit})"
        if entry.get('muted'):
            title += " — MUTED"
        self.impact_editor_label_var.set(title)

        def _fmt(v):
            if v is None or v == "":
                return ""
            try:
                f = float(v)
            except (TypeError, ValueError):
                return ""
            return ("%g" % f)

        self.impact_value_min_var.set(_fmt(entry.get('value_min')))
        self.impact_value_max_var.set(_fmt(entry.get('value_max')))
        self.impact_observable_var.set(bool(entry.get('observable', False)))
        # Populate the spawn-strategy editor from the impact's spawn block
        # (project override). Missing/invalid blocks fall back to uniform.
        spawn_cfg = entry.get('spawn') if isinstance(entry, dict) else None
        self._populate_spawn_editor(self.impact_spawn_vars,
                                    spawn_cfg if isinstance(spawn_cfg, dict) else {})
        # Show/hide the Spawn Strategy frame based on the observable flag.
        self._sync_impact_spawn_visibility()
        self._set_impact_editor_enabled(not entry.get('muted'))

    def apply_impact_changes(self):
        entry = self._selected_impact_entry()
        if entry is None:
            return
        if entry.get('muted'):
            messagebox.showinfo("Muted",
                                "This impact is muted. Unmute it before editing.")
            return
        mn_s = self.impact_value_min_var.get().strip()
        mx_s = self.impact_value_max_var.get().strip()
        if mn_s == "" or mx_s == "":
            messagebox.showwarning("Missing value",
                                   "Both Min and Max must be set.")
            return
        try:
            mn = float(mn_s)
            mx = float(mx_s)
        except ValueError:
            messagebox.showwarning("Invalid value",
                                   "Min and Max must be non-negative numbers.")
            return
        if mn < 0 or mx < 0 or mx < mn:
            messagebox.showwarning("Invalid range",
                                   "Require 0 <= Min <= Max.")
            return
        new_observable = bool(self.impact_observable_var.get())
        old_observable = bool(entry.get('observable', False))
        if new_observable != old_observable:
            display = self.global_library.get("impact_definitions", {}).get(
                entry.get('impact_id', ''), {}).get(
                    "display_name", entry.get('impact_id', ''))
            action = "added to" if new_observable else "removed from"
            messagebox.showinfo(
                "Observation data changed",
                f"'{display}' will be {action} the policy network's "
                f"observation input. This changes the input layer width of "
                f"every decision-maker policy network, so existing "
                f"checkpoints become incompatible and the policies must be "
                f"retrained from scratch. Save the project for the change "
                f"to take effect.")
        # Collect the spawn-strategy block strictly so unparseable entries
        # abort Apply (mirrors the per-FG spawn editor behaviour). The
        # block is only persisted for non-uniform modes or when explicit
        # parameters are present; uniform with no params is the default
        # and is omitted to keep YAML diffs minimal.
        spawn_errors = []
        spawn_dict = self._collect_spawn_dict(self.impact_spawn_vars,
                                              errors=spawn_errors)
        if spawn_errors:
            messagebox.showerror(
                "Invalid spawn parameters",
                "Could not apply impact changes:\n\n" + "\n".join(spawn_errors))
            return

        entry['value_min'] = mn
        entry['value_max'] = mx
        if new_observable:
            entry['observable'] = True
        else:
            entry.pop('observable', None)
        # Persist spawn block. Drop the entry entirely when the mode is
        # uniform with no extra params (legacy i.i.d. uniform sampling).
        if isinstance(spawn_dict, dict) and spawn_dict:
            mode = spawn_dict.get('mode', 'uniform')
            extra_keys = [k for k in spawn_dict.keys() if k != 'mode']
            if mode == 'uniform' and not extra_keys:
                entry.pop('spawn', None)
            else:
                entry['spawn'] = spawn_dict
        else:
            entry.pop('spawn', None)
        self._mark_dirty()
        # Refresh the listbox so the [observable] suffix tracks the
        # checkbox state. Preserve the current selection so the editor
        # doesn't deselect on Apply.
        try:
            sel = self.impact_listbox.curselection()
            self.update_impact_list()
            if sel:
                self.impact_listbox.selection_clear(0, "end")
                self.impact_listbox.selection_set(sel[0])
                self.impact_listbox.activate(sel[0])
                self.on_impact_select()
        except tk.TclError:
            pass

    def setup_matrix_tab(self):
        # Scrollable container for FG Interactions tab
        self.matrix_canvas = tk.Canvas(self.matrix_tab)
        self.matrix_scrollbar = ttk.Scrollbar(self.matrix_tab, orient="vertical", command=self.matrix_canvas.yview)
        self.matrix_container = ttk.Frame(self.matrix_canvas)

        self.matrix_container.bind(
            "<Configure>",
            lambda e: self.matrix_canvas.configure(
                scrollregion=self.matrix_canvas.bbox("all")
            )
        )

        self.matrix_canvas.create_window((0, 0), window=self.matrix_container, anchor="nw")
        self.matrix_hscrollbar = ttk.Scrollbar(self.matrix_tab, orient="horizontal", command=self.matrix_canvas.xview)
        self.matrix_canvas.configure(
            yscrollcommand=self.matrix_scrollbar.set,
            xscrollcommand=self.matrix_hscrollbar.set,
        )

        self.matrix_hscrollbar.pack(side="bottom", fill="x")
        self.matrix_scrollbar.pack(side="right", fill="y")
        self.matrix_canvas.pack(side="left", expand=True, fill="both")

        # Scrollable container for Impact Interactions tab
        self.impact_canvas = tk.Canvas(self.impact_tab)
        self.impact_scrollbar = ttk.Scrollbar(self.impact_tab, orient="vertical", command=self.impact_canvas.yview)
        self.impact_container = ttk.Frame(self.impact_canvas)

        self.impact_container.bind(
            "<Configure>",
            lambda e: self.impact_canvas.configure(
                scrollregion=self.impact_canvas.bbox("all")
            )
        )

        self.impact_canvas.create_window((0, 0), window=self.impact_container, anchor="nw")
        self.impact_hscrollbar = ttk.Scrollbar(self.impact_tab, orient="horizontal", command=self.impact_canvas.xview)
        self.impact_canvas.configure(
            yscrollcommand=self.impact_scrollbar.set,
            xscrollcommand=self.impact_hscrollbar.set,
        )

        self.impact_hscrollbar.pack(side="bottom", fill="x")
        self.impact_scrollbar.pack(side="right", fill="y")
        self.impact_canvas.pack(side="left", expand=True, fill="both")

        # Bind mouse wheel scrolling (vertical) and Shift+wheel (horizontal)
        self.matrix_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.matrix_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.matrix_canvas.bind_all("<Button-5>", self._on_mousewheel)
        self.matrix_canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.matrix_canvas.bind_all("<Shift-Button-4>", self._on_shift_mousewheel)
        self.matrix_canvas.bind_all("<Shift-Button-5>", self._on_shift_mousewheel)
        
        self.refresh_matrix()

    def setup_inference_tab(self):
        """Build the Inference tab: per-FG fixed initial biomass (ton) used by inference.py."""
        self.inference_canvas = tk.Canvas(self.inference_tab, highlightthickness=0)
        self.inference_vscroll = ttk.Scrollbar(self.inference_tab, orient="vertical",
                                               command=self.inference_canvas.yview)
        self.inference_inner = ttk.Frame(self.inference_canvas)
        self.inference_inner.bind(
            "<Configure>",
            lambda e: self.inference_canvas.configure(scrollregion=self.inference_canvas.bbox("all"))
        )
        self.inference_canvas.create_window((0, 0), window=self.inference_inner, anchor="nw")
        self.inference_canvas.configure(yscrollcommand=self.inference_vscroll.set)
        self.inference_vscroll.pack(side="right", fill="y")
        self.inference_canvas.pack(side="left", expand=True, fill="both")

        # Catch-all DnD registration: when the user drops onto the scrollable
        # canvas (which sits between the row widgets and the toplevel window),
        # tkinterdnd2 reports the drop here instead of on the inner widgets.
        # We forward it to ``_inference_on_drop(None, data)`` which figures out
        # the target impact via the pointer location.
        if self._dnd_available():
            # Register catch-all DnD on the scrollable canvas, the inner
            # frame, and the toplevel root window. Drops sometimes land on
            # the canvas window item or even the bare toplevel (especially
            # over padding/empty areas), and without a toplevel-level
            # handler the <<Drop>> event is silently dropped by Tk.
            targets = [self.inference_canvas, self.inference_inner,
                       self.inference_tab, self.root]
            for w_ in targets:
                try:
                    w_.drop_target_register("DND_Files")  # type: ignore[attr-defined]
                    w_.dnd_bind("<<Drop>>",  # type: ignore[attr-defined]
                                lambda e: self._inference_on_drop(None, e.data))
                except Exception:
                    pass

        info = ttk.Label(
            self.inference_inner,
            text=("Initial Biomass (ton) per Functional Group, used exclusively by inference.py "
                  "when biomass is initially spawned spatially. No random sampling is performed."),
            wraplength=700, justify="left",
        )
        info.pack(padx=10, pady=(10, 5), anchor="w")

        self.inference_list_frame = ttk.LabelFrame(self.inference_inner, text="Initial Biomass per FG")
        self.inference_list_frame.pack(fill="x", padx=10, pady=5)

        # Per-FG StringVars indexed by group_id
        self.inference_vars = {}

        # --- Impact maps section ---
        impact_info = ttk.Label(
            self.inference_inner,
            text=("Impact maps used by inference.py. For each unmuted impact you can "
                  "supply an .npz file containing a single 2-D float array (the key "
                  "name does not matter). Drop the file onto any zone (requires the "
                  "'tkinterdnd2' package) or click to browse. "
                  "Missing maps are treated as an all-zero field at inference time. "
                  "Biomass and energy maps are still randomly spawned from the configured "
                  "initial biomass values."),
            wraplength=700, justify="left",
        )
        impact_info.pack(padx=10, pady=(10, 5), anchor="w")

        self.inference_impact_frame = ttk.LabelFrame(
            self.inference_inner, text="Impact Maps (.npz) per Impact Variable")
        self.inference_impact_frame.pack(fill="x", padx=10, pady=5)

        # Per-impact GUI state: impact_id -> dict(path_var, status_var, thumb_label, drop_zone, ...)
        self.inference_impact_widgets = {}
        # Keep PhotoImage refs alive (Tk garbage-collects otherwise).
        self._inference_thumb_refs = {}

        ttk.Button(self.inference_inner, text="Apply Inference Settings",
                   command=self.apply_inference_changes).pack(pady=10)

        self.refresh_inference_tab()

    def refresh_inference_tab(self):
        """Rebuild the per-FG input list to mirror the current project FGs."""
        if not hasattr(self, "inference_list_frame"):
            return
        for w in self.inference_list_frame.winfo_children():
            w.destroy()
        self.inference_vars = {}

        # Live validator: non-negative integer only (or empty during editing).
        def _is_nonneg_int(s):
            return s == "" or s.isdigit()
        vcmd = (self.inference_list_frame.register(_is_nonneg_int), "%P")

        entries = []
        for cat in ("decision_makers", "non_decision_makers"):
            for fg in self.project_data.get(cat, []) or []:
                entries.append((cat, fg))
        if not entries:
            ttk.Label(self.inference_list_frame,
                      text="(No FGs in project — add some in 'Project & FGs'.)").grid(
                row=0, column=0, padx=10, pady=10, sticky="w")
            return

        ttk.Label(self.inference_list_frame, text="Functional Group",
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=4)
        ttk.Label(self.inference_list_frame, text="Initial Biomass (ton)",
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky="w", padx=5, pady=4)

        # Sort alphabetically by display name within each category, DMs first.
        def _sort_key(item):
            cat, fg = item
            return (0 if cat == "decision_makers" else 1,
                    self.fg_display(fg['group_id']).lower())
        entries.sort(key=_sort_key)

        for i, (cat, fg) in enumerate(entries, start=1):
            gid = fg['group_id']
            muted = self._is_fg_muted(fg)
            lbl_text = self.fg_display(gid, include_sv=True)
            if muted:
                lbl_text += "  (muted)"
            lbl = tk.Label(self.inference_list_frame, text=lbl_text)
            if muted:
                lbl.configure(foreground=self.MUTED_FG_COLOR)
            lbl.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar()
            cur = fg.get("inference_initial_biomass")
            if cur is not None and cur != "":
                try:
                    var.set(str(int(round(float(cur)))))
                except (TypeError, ValueError):
                    var.set("")
            ent = ttk.Entry(self.inference_list_frame, textvariable=var, width=14,
                            validate="key", validatecommand=vcmd)
            ent.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            if muted:
                ent.configure(state="disabled")
            self.inference_vars[gid] = var

        # Rebuild the impact-map zones as well.
        self._refresh_inference_impact_maps()

    # ------------------------------------------------------------------
    # Inference impact maps (.npz loading + thumbnail preview)
    # ------------------------------------------------------------------
    def _dnd_available(self):
        """Return True if the Tk root supports tkinterdnd2 drop targets."""
        if getattr(self, "_dnd_available_cached", None) is not None:
            return self._dnd_available_cached
        ok = False
        try:
            import tkinterdnd2  # noqa: F401
            # ``drop_target_register`` is only injected onto widgets when the
            # root window was created via ``TkinterDnD.Tk()``.
            ok = hasattr(self.root, "TkdndVersion") or hasattr(
                tk.Frame(self.root), "drop_target_register")
        except Exception:
            ok = False
        self._dnd_available_cached = bool(ok)
        return self._dnd_available_cached

    def _refresh_inference_impact_maps(self):
        """Rebuild the per-impact .npz drop-zone rows under the Inference tab."""
        if not hasattr(self, "inference_impact_frame"):
            return
        for w in self.inference_impact_frame.winfo_children():
            w.destroy()
        self.inference_impact_widgets = {}

        ivs = [iv for iv in self.project_data.get('impact_variables', []) or []
               if isinstance(iv, dict) and not iv.get('muted')]
        if not ivs:
            ttk.Label(self.inference_impact_frame,
                      text="(No unmuted impact variables — add some in 'Project & FGs'.)").grid(
                row=0, column=0, padx=10, pady=10, sticky="w")
            return

        # Sort alphabetically by display name.
        def _disp(iv):
            iid = iv.get('impact_id', '')
            return self.global_library.get("impact_definitions", {}).get(
                iid, {}).get("display_name", iid)
        ivs.sort(key=lambda iv: _disp(iv).lower())

        stored = (self.project_data.get('inference', {}) or {}).get('impact_maps', {}) or {}

        # Configure consistent column widths so each row's name / zone /
        # thumbnail / buttons line up vertically across all impacts.
        self.inference_impact_frame.columnconfigure(0, minsize=160)
        self.inference_impact_frame.columnconfigure(1, minsize=330)
        self.inference_impact_frame.columnconfigure(2, minsize=90)
        self.inference_impact_frame.columnconfigure(3, minsize=80)

        for i, iv in enumerate(ivs):
            iid = iv['impact_id']
            disp = _disp(iv)

            # A transparent ``row_frame`` is kept only as a logical handle for
            # DnD registration (covers the whole row's area). The actual
            # visible widgets are gridded directly into
            # ``inference_impact_frame`` so their columns align across rows.
            row_frame = self.inference_impact_frame

            ttk.Label(row_frame, text=disp,
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=0, sticky="w", padx=(5, 8), pady=4)

            # Drop / browse zone — a sunken frame with status text.
            zone = tk.Frame(row_frame, relief="groove", borderwidth=2,
                            width=320, height=70, bg="#f4f4f4")
            zone.grid(row=i, column=1, sticky="w", padx=(0, 10), pady=4)
            zone.grid_propagate(False)
            dnd_active = self._dnd_available()
            default_msg = ("Drop .npz here or click to browse"
                           if dnd_active else "Click to browse (.npz)")
            status_var = tk.StringVar(value=default_msg)
            status_lbl = tk.Label(zone, textvariable=status_var, bg="#f4f4f4",
                                  wraplength=300, justify="center")
            status_lbl.place(relx=0.5, rely=0.5, anchor="center")

            # Bind click on the zone itself for browsing.
            def _browse(_e=None, k=iid):
                self._inference_browse_map(k)
            zone.bind("<Button-1>", _browse)
            status_lbl.bind("<Button-1>", _browse)

            # Register the zone as a DnD drop target if the root is a
            # TkinterDnD-enabled window. Without that the <<Drop>> event
            # will never fire, so we silently skip registration and the
            # user can still click to browse.
            # Defer DnD registration until after thumb_lbl/btns exist so we can
            # register the whole row — see below.
            dnd_widgets_to_register = [zone, status_lbl]

            # Thumbnail preview (PhotoImage installed on demand).
            thumb_lbl = tk.Label(row_frame, bg="#ffffff", width=8, height=4,
                                 relief="solid", borderwidth=1)
            thumb_lbl.grid(row=i, column=2, sticky="w", padx=(0, 10), pady=4)

            btns = ttk.Frame(row_frame)
            btns.grid(row=i, column=3, sticky="w", pady=4)
            ttk.Button(btns, text="Clear",
                       command=lambda k=iid: self._inference_clear_map(k)).pack(side="left")

            self.inference_impact_widgets[iid] = {
                'status_var': status_var,
                'thumb_lbl': thumb_lbl,
                'zone': zone,
                'path': None,
            }

            # Register DnD on every widget in the row so the <<Drop>> event
            # fires no matter where on the row the user releases the file.
            # This is necessary because the row lives inside a scrollable
            # Canvas, where DnD events on inner widgets can otherwise be
            # swallowed by the canvas window item.
            if dnd_active:
                for w_ in dnd_widgets_to_register + [thumb_lbl, row_frame, btns]:
                    try:
                        w_.drop_target_register("DND_Files")  # type: ignore[attr-defined]
                        w_.dnd_bind("<<Drop>>",  # type: ignore[attr-defined]
                                    lambda e, k=iid: self._inference_on_drop(k, e.data))
                    except Exception:
                        pass

            # Load any previously stored path.
            stored_path = stored.get(iid)
            if stored_path:
                resolved = self._resolve_inference_map_path(stored_path)
                self._inference_set_map(iid, resolved, persist=False, store_path=stored_path)

    def _inference_browse_map(self, impact_id):
        path = filedialog.askopenfilename(
            title=f"Select .npz file for '{impact_id}'",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        if path:
            self._inference_set_map(impact_id, path, persist=True)

    def _inference_clear_map(self, impact_id):
        w = self.inference_impact_widgets.get(impact_id)
        if w is None:
            return
        w['path'] = None
        w['status_var'].set("Drop .npz here or click 'Browse…'")
        w['thumb_lbl'].configure(image='', text='')
        self._inference_thumb_refs.pop(impact_id, None)
        # Remove from project_data
        infer = self.project_data.setdefault('inference', {})
        maps = infer.setdefault('impact_maps', {})
        maps.pop(impact_id, None)
        if not maps:
            infer.pop('impact_maps', None)
        if not infer:
            self.project_data.pop('inference', None)

    def _inference_impact_under_pointer(self):
        """Return the impact_id whose row currently contains the mouse pointer.

        Used as a fallback when DnD is registered on a shared parent widget
        (canvas/root) rather than on each row. Walks up the widget hierarchy
        from the widget under the pointer until it finds one of the per-row
        zones registered in ``inference_impact_widgets``.
        """
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
            w = self.root.winfo_containing(x, y)
        except Exception:
            return None
        # Build reverse lookup: widget -> impact_id.
        zone_to_iid = {}
        for iid, info in self.inference_impact_widgets.items():
            zone = info.get('zone')
            if zone is not None:
                zone_to_iid[str(zone)] = iid
                # Also map all descendants of the zone's parent row_frame.
                parent = zone.master
                if parent is not None:
                    zone_to_iid[str(parent)] = iid
                    for child in parent.winfo_children():
                        zone_to_iid[str(child)] = iid
        cur = w
        while cur is not None:
            key = str(cur)
            if key in zone_to_iid:
                return zone_to_iid[key]
            try:
                cur = cur.master
            except Exception:
                break
        return None

    def _inference_on_drop(self, impact_id, data):
        """Handle tkinterdnd2 drop event (best-effort).

        ``impact_id`` may be ``None`` when the drop was registered on a shared
        widget (canvas/root); in that case we locate the row under the pointer.
        """
        if not data:
            return
        if impact_id is None:
            impact_id = self._inference_impact_under_pointer()
            if impact_id is None:
                return
        # tkinterdnd2 passes a brace-quoted, space-separated list of paths.
        paths = []
        cur, in_brace = [], False
        for ch in data:
            if ch == '{':
                in_brace = True
                continue
            if ch == '}':
                in_brace = False
                paths.append(''.join(cur))
                cur = []
                continue
            if ch == ' ' and not in_brace:
                if cur:
                    paths.append(''.join(cur))
                    cur = []
                continue
            cur.append(ch)
        if cur:
            paths.append(''.join(cur))
        if paths:
            self._inference_set_map(impact_id, paths[0], persist=True)

    def _resolve_inference_map_path(self, stored):
        """Resolve a stored (possibly relative) path against the project file dir."""
        if os.path.isabs(stored):
            return stored
        base = os.path.dirname(self.project_path) if self.project_path else os.getcwd()
        return os.path.normpath(os.path.join(base, stored))

    def _format_inference_map_path(self, abs_path):
        """Return path stored in YAML: relative to project dir if inside, else absolute."""
        if not self.project_path:
            return abs_path
        proj_dir = os.path.dirname(os.path.abspath(self.project_path))
        ap = os.path.abspath(abs_path)
        try:
            common = os.path.commonpath([proj_dir, ap])
        except ValueError:
            return ap
        if common == proj_dir:
            return os.path.relpath(ap, proj_dir)
        return ap

    def _inference_set_map(self, impact_id, path, persist=True, store_path=None):
        """Validate the .npz, update GUI (status + thumbnail), and (optionally) persist."""
        import numpy as _np
        w = self.inference_impact_widgets.get(impact_id)
        if w is None:
            return
        if not path or not os.path.isfile(path):
            messagebox.showwarning(
                "File not found",
                f"Could not open file:\n{path}")
            w['status_var'].set("Drop .npz here or click 'Browse…'")
            return
        if not path.lower().endswith('.npz'):
            messagebox.showwarning(
                "Wrong file type",
                f"Only .npz files are accepted (got: {os.path.basename(path)}).")
            return
        try:
            with _np.load(path, allow_pickle=False) as data:
                keys = list(data.files)
                arr = None
                # Priority 1: exact impact_id match.
                if impact_id in keys:
                    arr = data[impact_id]
                else:
                    # Priority 2: any 2-D array in the archive. Prefer the
                    # first one; ignore non-2D arrays (e.g. metadata vectors).
                    for k in keys:
                        cand = data[k]
                        if hasattr(cand, 'ndim') and cand.ndim == 2:
                            arr = cand
                            break
                if arr is None:
                    messagebox.showwarning(
                        "No 2-D array in .npz",
                        f"{os.path.basename(path)} does not contain any 2-D array "
                        f"that can be used as an impact map.\nKeys present: {keys}")
                    return
                arr = _np.asarray(arr)
        except Exception as e:
            messagebox.showwarning("Invalid .npz",
                                   f"Could not read '{impact_id}' from {os.path.basename(path)}:\n{e}")
            return
        if arr.ndim != 2:
            messagebox.showwarning(
                "Wrong shape",
                f"Array '{impact_id}' must be 2-D (got shape {arr.shape}).")
            return
        try:
            arr_f = arr.astype('float32', copy=False)
        except Exception as e:
            messagebox.showwarning("Invalid dtype",
                                   f"Could not convert array to float32: {e}")
            return
        if not _np.isfinite(arr_f).all():
            messagebox.showwarning(
                "Non-finite values",
                f"Array '{impact_id}' contains NaN or inf values; please clean before use.")
            return

        # Optional sanity-check against the per-impact [value_min, value_max].
        ie = self._find_impact_entry(impact_id) or {}
        vmin = ie.get('value_min')
        vmax = ie.get('value_max')
        try:
            vmin_f = float(vmin) if vmin is not None and vmin != "" else None
            vmax_f = float(vmax) if vmax is not None and vmax != "" else None
        except (TypeError, ValueError):
            vmin_f = vmax_f = None
        amin, amax = float(arr_f.min()), float(arr_f.max())
        warn_range = ""
        if vmin_f is not None and amin < vmin_f - 1e-9:
            warn_range = f"  ⚠ min={amin:g} below value_min={vmin_f:g}"
        if vmax_f is not None and amax > vmax_f + 1e-9:
            warn_range = (warn_range or "") + f"  ⚠ max={amax:g} above value_max={vmax_f:g}"

        # Update GUI
        w['path'] = os.path.abspath(path)
        H, W_ = arr_f.shape
        w['status_var'].set(
            f"{os.path.basename(path)}\nshape={H}×{W_}  "
            f"min={amin:g}  max={amax:g}{warn_range}")
        self._render_inference_thumbnail(impact_id, arr_f)

        # Persist into project_data (relative path if inside the project dir).
        if persist:
            stored = self._format_inference_map_path(w['path'])
            infer = self.project_data.setdefault('inference', {})
            maps = infer.setdefault('impact_maps', {})
            maps[impact_id] = stored
        elif store_path is not None:
            # Keep whatever path was loaded from disk (no rewrite).
            pass

    def _render_inference_thumbnail(self, impact_id, arr):
        """Render a small grayscale thumbnail of ``arr`` (min/max normalised for display only)."""
        try:
            from PIL import Image, ImageTk
        except Exception:
            return
        a = arr.astype('float32', copy=False)
        amin = float(a.min())
        amax = float(a.max())
        if amax > amin:
            disp = (a - amin) / (amax - amin)
        else:
            disp = a * 0.0
        disp = (disp * 255.0).clip(0, 255).astype('uint8')
        img = Image.fromarray(disp, mode='L')
        # Fit into ~80×64 pixels (preserve aspect).
        img.thumbnail((80, 64))
        photo = ImageTk.PhotoImage(img)
        w = self.inference_impact_widgets.get(impact_id)
        if w is None:
            return
        w['thumb_lbl'].configure(image=photo, text='', width=img.width, height=img.height)
        self._inference_thumb_refs[impact_id] = photo  # prevent GC

    def apply_inference_changes(self):
        """Persist the inference-tab values onto the project FG entries."""
        # Build a lookup from group_id to its entry across both categories.
        for cat in ("decision_makers", "non_decision_makers"):
            for fg in self.project_data.get(cat, []) or []:
                gid = fg.get('group_id')
                if gid is None or gid not in self.inference_vars:
                    continue
                if self._is_fg_muted(fg):
                    # Skip muted FGs entirely — their stored value is preserved.
                    continue
                raw = self.inference_vars[gid].get()
                if raw == "":
                    fg.pop("inference_initial_biomass", None)
                    continue
                try:
                    v = int(raw)
                    if v < 0:
                        v = 0
                except ValueError:
                    fg.pop("inference_initial_biomass", None)
                    continue
                fg["inference_initial_biomass"] = v
        self._mark_dirty()
        messagebox.showinfo(
            "Success",
            "Inference initial biomass updated on project entries. Remember to Save Project."
        )

    def refresh_matrix(self):
        for widget in self.matrix_container.winfo_children():
            widget.destroy()
        for widget in self.impact_container.winfo_children():
            widget.destroy()

        fgs = sorted(self._all_fg_ids(), key=lambda fg_id: self.fg_display(fg_id).lower())
        # Row IDs are restricted to decision-making FGs: non-decision-makers
        # have no policy/interaction choices to configure, so they only ever
        # appear as columns (potential prey / observable / impact targets).
        dm_ids = {fg['group_id']
                  for fg in (self.project_data.get('decision_makers', []) or [])}
        dm_fgs = [fid for fid in fgs if fid in dm_ids]
        if not fgs:
            ttk.Label(self.matrix_container, text="Add FGs to the project to see interaction matrices.").pack(padx=10, pady=10)
            ttk.Label(self.impact_container, text="Add FGs to the project to see impact interactions.").pack(padx=10, pady=10)
            return

        self.matrix_entries = {}
        self.matrix_widgets = {}
        self.matrix_tables = {}
        self.matrix_col_headers = {}
        self.matrix_row_headers = {}
        
        # FG Interactions tab: Predation only. The previous per-(predator, prey)
        # ``max_intake_rate`` matrix (I_XY) has been removed; the value is now a
        # general per-predator property, edited in the FG Editor on the
        # "Project & FGs" tab and stored in ``species_definitions``.
        self.create_matrix_section("Predation (row eats column)", "preys_on", dm_fgs, fgs, cell_type="bool",
                                   parent=self.matrix_container)

        # Observability matrix: definierar vilka FGs varje FG kan observera i
        # sitt input space. Muteade rader/kolumner gråmarkeras och
        # deaktiveras precis som i predationsmatrisen. Om en observerad FG
        # är muteed matas 0 in i policynätverket vid träning/inferens.
        # Placerad direkt under predationsmatrisen eftersom de är tätt
        # kopplade (preys_on => observes forceras True).
        self.create_matrix_section(
            "Observability (row observes column)", "observes",
            dm_fgs, fgs, cell_type="bool", parent=self.matrix_container)

        # Assimilation Factor matrix: fraktion av bytets energiinnehåll
        # som faktiskt tas upp vid predation. Värden i [0, 1], default 1.0.
        # Dynamiskt kopplad till predationsmatrisen: en cell är aktiv
        # endast om motsvarande preys_on=True.
        self.create_matrix_section(
            "Assimilation Factor (row eats column)", "assimilation_factor",
            dm_fgs, fgs, cell_type="unit_slider", parent=self.matrix_container)

        # Handling time matrix (Holling Type II 'h'). Non-negativ float,
        # default 0.0 (= pure Type I). Cell aktiv endast om preys_on=True.
        # Dold på användarens begäran — värdena bevaras i library YAML
        # och i interaction_definitions, men matrisen renderas inte i
        # GUI:t. Avkommentera raderna nedan för att återaktivera.
        # self.create_matrix_section(
        #     "Handling time (row eats column)", "handling_time",
        #     dm_fgs, fgs, cell_type="nonneg_float", parent=self.matrix_container)

        # Impact Interactions tab: Impact Affects (boolean) and Impact Tables (table editor per cell)
        impacts = [iv['impact_id'] for iv in self.project_data.get('impact_variables', [])]
        impacts.sort(key=lambda imp: self.global_library.get("impact_definitions", {}).get(imp, {}).get("display_name", imp).lower())
        if impacts:
            self.create_matrix_section("Impact Affects (row impacted by column)", "impact_affects", dm_fgs, impacts,
                                       cell_type="bool", parent=self.impact_container)
            self.create_matrix_section("Impact Tables (value, biomass factor, energy factor)", "impact_table", dm_fgs, impacts,
                                       cell_type="table", parent=self.impact_container)
        else:
            ttk.Label(self.impact_container, text="Add impact variables to the project to see impact interactions.").pack(padx=10, pady=10)


        # Link predation checkboxes to assimilation_factor entry+slider enable-state.
        # x preys on y = false => motsvarande fg-par i assimilationsmatrisen deaktiveras.
        for key, data in self.matrix_entries.items():
            preys_var = data.get("preys_on")
            if preys_var is None:
                continue
            for dep_key in ("assimilation_factor", "handling_time"):
                dep_widget = self.matrix_widgets.get(key, {}).get(dep_key)
                if dep_widget is None:
                    continue
                def make_updater(var=preys_var, cell=dep_widget):
                    def update(*_):
                        enabled = bool(var.get())
                        target = "normal" if enabled else "disabled"
                        try:
                            cell.configure(state=target)
                        except tk.TclError:
                            pass
                        self._set_widget_tree_state(cell, enabled)
                    return update
                updater = make_updater()
                preys_var.trace_add("write", updater)
                updater()

        # Link predation checkboxes to observability matrix:
        # x preys on y  ==>  x observes y is forced True and the
        # observability checkbox is disabled (the user cannot uncheck
        # observability of a prey). When predation is cleared the
        # observability cell becomes editable again (its previous value
        # is preserved unless it was forced on by this link — in which
        # case it stays True so the user can manually clear it if
        # desired). Self-references and muted-row/col cases are handled
        # by their own disable-paths (later in this method) and are
        # still respected: they may add more disabled-state on top but
        # cannot override the True+disabled set here.
        for key, data in self.matrix_entries.items():
            preys_var = data.get("preys_on")
            if preys_var is None:
                continue
            if "_preys_on_" not in key:
                continue
            row_id, col_id = key.split("_preys_on_", 1)
            obs_key = f"{row_id}_observes_{col_id}"
            obs_entry = self.matrix_entries.get(obs_key, {})
            obs_var = obs_entry.get("observes")
            obs_widget = self.matrix_widgets.get(obs_key, {}).get("observes")
            if obs_var is None or obs_widget is None:
                continue
            def make_obs_updater(pvar=preys_var, ovar=obs_var, owid=obs_widget,
                                 r=row_id, c=col_id):
                def update(*_):
                    if bool(pvar.get()):
                        ovar.set(True)
                        try:
                            owid.configure(state="disabled")
                        except tk.TclError:
                            pass
                    else:
                        # Re-enable only if not a self-reference (which
                        # is permanently disabled by create_matrix_section).
                        if r != c:
                            try:
                                owid.configure(state="normal")
                            except tk.TclError:
                                pass
                return update
            obs_updater = make_obs_updater()
            preys_var.trace_add("write", obs_updater)
            obs_updater()

        # Link impact_affects checkboxes to impact_table button enable-state
        for key, data in self.matrix_entries.items():
            affects_var = data.get("impact_affects")
            if affects_var is None:
                continue
            table_widget = self.matrix_widgets.get(key, {}).get("impact_table")
            if table_widget is None:
                continue
            def make_impact_updater(var=affects_var, widget=table_widget):
                def update(*_):
                    widget.configure(state="normal" if var.get() else "disabled")
                return update
            updater = make_impact_updater()
            affects_var.trace_add("write", updater)
            updater()

        # Disable widgets belonging to muted FGs (rows and columns) or muted
        # impact variables (columns). Values stay visible but cannot be edited.
        muted_fgs = {fid for fid in self._all_fg_ids() if self._is_fg_id_muted(fid)}
        muted_impacts = {iv['impact_id'] for iv in self.project_data.get('impact_variables', []) or []
                         if self._is_impact_muted(iv)}
        for key, widgets in self.matrix_widgets.items():
            # Parse row/col from the key.
            if "_observes_" in key:
                row_id, col_id = key.split("_observes_", 1)
                disable = row_id in muted_fgs or col_id in muted_fgs
            elif "_preys_on_" in key:
                row_id, col_id = key.split("_preys_on_", 1)
                disable = row_id in muted_fgs or col_id in muted_fgs
            elif "_impacted_by_" in key:
                row_id, col_id = key.split("_impacted_by_", 1)
                disable = row_id in muted_fgs or col_id in muted_impacts
            else:
                disable = False
            if not disable:
                continue
            for w in widgets.values():
                try:
                    w.configure(state="disabled")
                except tk.TclError:
                    # Container widgets (ttk.Frame) don't support 'state';
                    # disable all leaf children recursively instead.
                    self._set_widget_tree_state(w, False)

        # Lighter header text for muted FGs/impacts (rows and columns).
        for row_id, labels in self.matrix_row_headers.items():
            color = self.MUTED_FG_COLOR if row_id in muted_fgs else "black"
            for lbl in labels:
                try:
                    lbl.configure(foreground=color)
                except tk.TclError:
                    pass
        for (kind, col_id), labels in self.matrix_col_headers.items():
            if kind == "impact":
                muted = col_id in muted_impacts
            else:
                muted = col_id in muted_fgs
            color = self.MUTED_FG_COLOR if muted else "black"
            for lbl in labels:
                try:
                    lbl.configure(foreground=color)
                except tk.TclError:
                    pass

        ttk.Button(self.matrix_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)
        if impacts:
            ttk.Button(self.impact_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)

    def create_matrix_section(self, title, data_key, row_ids, col_ids, cell_type="entry", parent=None):
        if parent is None:
            parent = self.matrix_container
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="x", padx=10, pady=10)

        impact_keys = ("impact_affects", "impact_table")
        # Track header labels so refresh_matrix can apply muted styling
        # (lighter foreground) to entire row/column header text.
        if not hasattr(self, "matrix_col_headers"):
            self.matrix_col_headers = {}
        if not hasattr(self, "matrix_row_headers"):
            self.matrix_row_headers = {}
        # Headers — use tk.Label so we can recolour per-label.
        tk.Label(frame, text="Group \\ Var").grid(row=0, column=0, padx=5, pady=5)
        for j, col_id in enumerate(col_ids):
            # Use display_name from impact_definitions if available
            if data_key in impact_keys:
                label_text = self.global_library.get("impact_definitions", {}).get(col_id, {}).get("display_name", col_id)
            else:
                label_text = self.fg_display(col_id)
            hdr = tk.Label(frame, text=label_text)
            hdr.grid(row=0, column=j+1, padx=5, pady=5)
            kind = "impact" if data_key in impact_keys else "fg"
            self.matrix_col_headers.setdefault((kind, col_id), []).append(hdr)

        for i, row_id in enumerate(row_ids):
            row_label = self.fg_display(row_id)
            rhdr = tk.Label(frame, text=row_label)
            rhdr.grid(row=i+1, column=0, padx=5, pady=5)
            self.matrix_row_headers.setdefault(row_id, []).append(rhdr)
            for j, col_id in enumerate(col_ids):
                # Unique key for storage
                if data_key in impact_keys:
                    key = f"{row_id}_impacted_by_{col_id}"
                elif data_key == "observes":
                    key = f"{row_id}_observes_{col_id}"
                else:
                    key = f"{row_id}_preys_on_{col_id}"
                
                if key not in self.matrix_entries:
                    self.matrix_entries[key] = {}

                # Look for existing value in library
                existing = self.global_library.get("interaction_definitions", {}).get(key, {})
                if cell_type == "entry":
                    _default = ""
                elif cell_type == "unit_slider":
                    _default = 1.0
                elif cell_type == "nonneg_float":
                    _default = 0.0
                else:
                    _default = False
                val = existing.get(data_key, _default)

                if cell_type == "bool":
                    # Self-reference in the Observability matrix: a DM
                    # always observes itself (B_own/E_own are part of the
                    # center observation by construction). Force the cell
                    # to True and disable it so the user cannot toggle it
                    # off. This is independent of mute state.
                    force_self_observe = (data_key == "observes" and row_id == col_id)
                    if force_self_observe:
                        val = True
                    var = tk.BooleanVar(value=bool(val))
                    # Use classic tk.Checkbutton (not ttk) because its
                    # indicator visibly dims when state="disabled", which
                    # makes muted rows/columns clearly inactive even when
                    # the box is checked. ttk.Checkbutton's indicator is
                    # theme-controlled and often looks identical when
                    # checked+disabled vs checked+normal.
                    widget = tk.Checkbutton(frame, variable=var,
                                            disabledforeground=self.MUTED_FG_COLOR)
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                    if force_self_observe:
                        widget.configure(state="disabled")
                elif cell_type == "unit_slider":
                    # Enkelt entry-fält med värde i [0, 1]. Default = 1.0
                    # (full assimilation). Live-validering: bara värden i
                    # [0, 1] accepteras i entry-fältet. (Sliders borttagna
                    # på användarens begäran.)
                    cur_val = val
                    if cur_val == "" or cur_val is None:
                        cur_val = 1.0
                    try:
                        cur_f = float(cur_val)
                    except (TypeError, ValueError):
                        cur_f = 1.0
                    cur_f = max(0.0, min(1.0, cur_f))
                    var = tk.StringVar(value=f"{cur_f:g}")

                    def _validate_unit(proposed):
                        if proposed in ("", "."):
                            return True
                        try:
                            v = float(proposed)
                        except ValueError:
                            return False
                        return 0.0 <= v <= 1.0
                    vcmd = (frame.register(_validate_unit), "%P")

                    widget = ttk.Entry(frame, textvariable=var, width=6,
                                       validate="key", validatecommand=vcmd)
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                elif cell_type == "nonneg_float":
                    # Icke-negativt float-fält (t.ex. handling_time).
                    # Default = 0.0. Live-validering: bara >=0 accepteras.
                    cur_val = val
                    if cur_val == "" or cur_val is None:
                        cur_val = 0.0
                    try:
                        cur_f = float(cur_val)
                    except (TypeError, ValueError):
                        cur_f = 0.0
                    if cur_f < 0.0:
                        cur_f = 0.0
                    var = tk.StringVar(value=f"{cur_f:g}")

                    def _validate_nonneg(proposed):
                        if proposed in ("", "."):
                            return True
                        try:
                            v = float(proposed)
                        except ValueError:
                            return False
                        return v >= 0.0
                    vcmd = (frame.register(_validate_nonneg), "%P")

                    widget = ttk.Entry(frame, textvariable=var, width=6,
                                       validate="key", validatecommand=vcmd)
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                elif cell_type == "table":
                    # Load existing table (list of dicts) from library, if any
                    existing_table = existing.get("impact_table")
                    if not isinstance(existing_table, list):
                        existing_table = []
                    if not hasattr(self, "matrix_tables"):
                        self.matrix_tables = {}
                    self.matrix_tables[key] = [dict(row) for row in existing_table]
                    var = None  # tables are not stored in a tk var
                    widget = ttk.Button(
                        frame, text="Edit Table…",
                        command=lambda k=key, r=row_id, c=col_id: self.open_impact_table_editor(k, r, c),
                    )
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                else:
                    var = tk.StringVar(value=str(val))
                    widget = ttk.Entry(frame, textvariable=var, width=10)
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                if var is not None:
                    self.matrix_entries[key][data_key] = var
                if not hasattr(self, "matrix_widgets"):
                    self.matrix_widgets = {}
                if key not in self.matrix_widgets:
                    self.matrix_widgets[key] = {}
                self.matrix_widgets[key][data_key] = widget

    # ------------------------------------------------------------------
    # Mute (soft-delete) helpers
    # ------------------------------------------------------------------
    # An FG or impact variable that is "muted" is semantically equivalent to
    # being removed from the project (it does not exist for train.py /
    # inference.py / config_loader), but its configuration is preserved so it
    # can be unmuted later without re-entering values. The muted flag lives on
    # the project FG/impact entry (per-project state), not in the library.

    MUTED_FG_COLOR = "gray60"

    # Display unit per impact_id (shown in parentheses after the impact's
    # display name in the project's Impact Variables list and after the
    # "Value" header in the impact-table editor under "Impact Interactions").
    IMPACT_UNITS = {
        "bottom_trawling": "MW-h/year",
        "pelagic_trawling": "MW-h/year",
        "windfarm_noise": "dB",
        "rotor": "fraction/cell",
    }

    def _impact_unit(self, impact_id):
        return self.IMPACT_UNITS.get(impact_id, "undefined")

    def _is_fg_muted(self, fg_entry):
        return bool(isinstance(fg_entry, dict) and fg_entry.get("muted"))

    def _is_impact_muted(self, iv_entry):
        return bool(isinstance(iv_entry, dict) and iv_entry.get("muted"))

    def _find_fg_entry(self, fg_id):
        for cat in ("decision_makers", "non_decision_makers"):
            for fg in self.project_data.get(cat, []) or []:
                if fg.get('group_id') == fg_id:
                    return fg
        return None

    def _find_impact_entry(self, impact_id):
        for iv in self.project_data.get('impact_variables', []) or []:
            if iv.get('impact_id') == impact_id:
                return iv
        return None

    def _is_fg_id_muted(self, fg_id):
        e = self._find_fg_entry(fg_id)
        return self._is_fg_muted(e) if e else False

    def _is_impact_id_muted(self, impact_id):
        e = self._find_impact_entry(impact_id)
        return self._is_impact_muted(e) if e else False

    def _apply_listbox_mute_styling(self):
        """Recolour listbox rows so muted entries appear in a lighter colour."""
        # DM list
        if hasattr(self, 'fg_listbox'):
            for i, fg in enumerate(self.project_data.get('decision_makers', []) or []):
                color = self.MUTED_FG_COLOR if self._is_fg_muted(fg) else ""
                try:
                    self.fg_listbox.itemconfig(i, foreground=color)
                except tk.TclError:
                    pass
        # NDM list
        if hasattr(self, 'ndm_listbox'):
            for i, fg in enumerate(self.project_data.get('non_decision_makers', []) or []):
                color = self.MUTED_FG_COLOR if self._is_fg_muted(fg) else ""
                try:
                    self.ndm_listbox.itemconfig(i, foreground=color)
                except tk.TclError:
                    pass
        # Impact list
        if hasattr(self, 'impact_listbox'):
            for i, iv in enumerate(self.project_data.get('impact_variables', []) or []):
                color = self.MUTED_FG_COLOR if self._is_impact_muted(iv) else ""
                try:
                    self.impact_listbox.itemconfig(i, foreground=color)
                except tk.TclError:
                    pass

    def _refresh_mute_button_labels(self):
        """Update Mute/Unmute button text to reflect current selection state."""
        def _label_for_fg(category, btn):
            if not hasattr(self, btn.__class__.__name__):
                pass
            listbox = self._listbox_for(category)
            sel = listbox.curselection()
            fgs = self.project_data.get(category, []) or []
            if not sel or sel[0] >= len(fgs):
                btn.configure(text="Mute", state="disabled")
                return
            muted = self._is_fg_muted(fgs[sel[0]])
            btn.configure(text="Unmute" if muted else "Mute", state="normal")

        if hasattr(self, 'dm_mute_btn'):
            _label_for_fg("decision_makers", self.dm_mute_btn)
        if hasattr(self, 'ndm_mute_btn'):
            _label_for_fg("non_decision_makers", self.ndm_mute_btn)
        if hasattr(self, 'impact_mute_btn'):
            sel = self.impact_listbox.curselection()
            ivs = self.project_data.get('impact_variables', []) or []
            if not sel or sel[0] >= len(ivs):
                self.impact_mute_btn.configure(text="Mute", state="disabled")
            else:
                muted = self._is_impact_muted(ivs[sel[0]])
                self.impact_mute_btn.configure(text="Unmute" if muted else "Mute", state="normal")

    def _set_widget_tree_state(self, widget, enabled):
        """Recursively enable/disable all leaf widgets that support a 'state' option."""
        target = "normal" if enabled else "disabled"
        for child in widget.winfo_children():
            try:
                child.configure(state=target)
            except tk.TclError:
                # Some containers (Frame, LabelFrame) don't support state — descend.
                pass
            self._set_widget_tree_state(child, enabled)

    def toggle_mute_fg(self, category):
        listbox = self._listbox_for(category)
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection",
                                   "Please select a Functional Group from the list.")
            return
        fgs = self.project_data.get(category, []) or []
        if sel[0] >= len(fgs):
            return
        fg_entry = fgs[sel[0]]
        fg_entry['muted'] = not self._is_fg_muted(fg_entry)
        if not fg_entry['muted']:
            # Remove the key entirely when active to keep YAML tidy.
            fg_entry.pop('muted', None)
        self._mark_dirty()
        # Refresh all views that depend on muted state.
        self._apply_listbox_mute_styling()
        self._refresh_mute_button_labels()
        self.refresh_matrix()
        if hasattr(self, 'refresh_inference_tab'):
            self.refresh_inference_tab()
        # If editor currently shows this FG, refresh its disabled state.
        if self.active_fg_category == category:
            self.on_fg_select(category)

    def toggle_mute_impact(self):
        sel = self.impact_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection",
                                   "Please select an Impact Variable from the list.")
            return
        ivs = self.project_data.get('impact_variables', []) or []
        if sel[0] >= len(ivs):
            return
        iv = ivs[sel[0]]
        iv['muted'] = not self._is_impact_muted(iv)
        if not iv['muted']:
            iv.pop('muted', None)
        self._mark_dirty()
        self._apply_listbox_mute_styling()
        self._refresh_mute_button_labels()
        self.refresh_matrix()
        # Refresh the Impact Editor so it enables/disables immediately
        # following a mute/unmute toggle on the currently selected impact.
        if hasattr(self, 'impact_editor_frame'):
            self.on_impact_select()
        # Muting/unmuting changes the set of impacts shown on the Inference tab.
        if hasattr(self, 'inference_impact_frame'):
            self._refresh_inference_impact_maps()

    def _listbox_for(self, category):
        return self.fg_listbox if category == "decision_makers" else self.ndm_listbox

    def _all_fg_entries(self):
        """Return list of (category, group_id) for every FG currently in the project."""
        out = []
        for cat in ("decision_makers", "non_decision_makers"):
            for fg in self.project_data.get(cat, []) or []:
                out.append((cat, fg['group_id']))
        return out

    def _all_fg_ids(self):
        return [gid for _, gid in self._all_fg_entries()]

    def _find_fg_category(self, fg_id):
        for cat in ("decision_makers", "non_decision_makers"):
            if any(fg['group_id'] == fg_id for fg in self.project_data.get(cat, []) or []):
                return cat
        return None

    def on_fg_select(self, category):
        listbox = self._listbox_for(category)
        selection = listbox.curselection()
        if not selection:
            return
        # Deselect the other listbox to avoid ambiguity
        other = self.ndm_listbox if category == "decision_makers" else self.fg_listbox
        other.selection_clear(0, "end")
        self.active_fg_category = category

        idx = selection[0]
        fgs = self.project_data.get(category, [])
        if idx >= len(fgs):
            return
        fg_entry = fgs[idx]
        fg_id = fg_entry['group_id']
        config = self.current_fg_configs.get(fg_id, {})
        is_muted = self._is_fg_muted(fg_entry)

        # initial_biomass range is a per-project FG override (not a library field).
        # Read min/max from the project FG entry, with backward compatibility for
        # legacy scalar `initial_biomass` (treated as min == max).
        init_min_val, init_max_val = self._read_initial_biomass_range(fg_entry, fg_id)

        # Show the editor matching the FG category, hide the other.
        if category == "decision_makers":
            self.ndm_editor_frame.pack_forget()
            if not self.editor_frame.winfo_ismapped():
                self.editor_frame.pack(fill="x", padx=10, pady=5)
            title = f"FG Editor ({self.fg_display(fg_id, include_sv=True)})"
            if is_muted:
                title += " — MUTED"
            self.editor_frame.configure(text=title)
            for key, var in self.prop_vars.items():
                if key == "initial_biomass_min":
                    var.set("" if init_min_val is None else str(init_min_val))
                    continue
                if key == "initial_biomass_max":
                    var.set("" if init_max_val is None else str(init_max_val))
                    continue
                val = config.get(key, "")
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(val))
                else:
                    var.set(str(val))
        else:
            self.editor_frame.pack_forget()
            if not self.ndm_editor_frame.winfo_ismapped():
                self.ndm_editor_frame.pack(fill="x", padx=10, pady=5)
            title = f"FG Editor ({self.fg_display(fg_id, include_sv=True)})"
            if is_muted:
                title += " — MUTED"
            self.ndm_editor_frame.configure(text=title)
            for key, var in self.ndm_prop_vars.items():
                if key == "initial_biomass_min":
                    var.set("" if init_min_val is None else str(init_min_val))
                    continue
                if key == "initial_biomass_max":
                    var.set("" if init_max_val is None else str(init_max_val))
                    continue
                val = config.get(key, "")
                var.set(str(val))

        # Populate the per-FG spawn editor from config['spawn'] (or library default).
        spawn_cfg = None
        if isinstance(config, dict):
            spawn_cfg = config.get("spawn")
        if not isinstance(spawn_cfg, dict):
            lib_entry = self.global_library.get("species_definitions", {}).get(fg_id, {})
            if isinstance(lib_entry, dict):
                spawn_cfg = lib_entry.get("spawn")
        spawn_store = self.spawn_vars if category == "decision_makers" else self.ndm_spawn_vars
        self._populate_spawn_editor(spawn_store, spawn_cfg or {})

        # Disable all editor widgets when the FG is muted (values are still
        # visible/read-only). Active FGs get the editor enabled normally.
        active_editor = self.editor_frame if category == "decision_makers" else self.ndm_editor_frame
        active_spawn = self.spawn_frame if category == "decision_makers" else self.ndm_spawn_frame
        self._set_widget_tree_state(active_editor, not is_muted)
        self._set_widget_tree_state(active_spawn, not is_muted)
        # Keep mute-button labels in sync with the freshly selected row.
        self._refresh_mute_button_labels()

    def apply_fg_changes(self):
        category = self.active_fg_category
        if category is None:
            messagebox.showwarning("No Selection", "Please select a Functional Group from the list.")
            return
        listbox = self._listbox_for(category)
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a Functional Group from the list.")
            return
        idx = selection[0]
        fgs = self.project_data.get(category, [])
        if idx >= len(fgs):
            return
        fg_id = fgs[idx]['group_id']
        if self._is_fg_muted(fgs[idx]):
            messagebox.showinfo(
                "Muted",
                "This FG is muted. Unmute it before editing its properties.",
            )
            return
        # Preserve existing display_name and is_decision_maker flag.
        # The flag is driven by which list the FG belongs to, not by a checkbox.
        existing = self.current_fg_configs.get(fg_id, {})
        display_name = existing.get("display_name") if isinstance(existing, dict) else None
        if not display_name:
            display_name = self.global_library.get("species_definitions", {}).get(fg_id, {}).get("display_name", fg_id)
        is_dm = (category == "decision_makers")
        # Start from existing config so we preserve fields not shown in the active editor.
        config = dict(existing) if isinstance(existing, dict) else {}
        config["display_name"] = display_name
        config["is_decision_maker"] = is_dm

        prop_vars = self.prop_vars if is_dm else self.ndm_prop_vars
        # initial_biomass range captured separately; stored per-project, not in library.
        initial_biomass_min_val = None
        initial_biomass_max_val = None
        for key, var in prop_vars.items():
            val = var.get()
            if key in ("initial_biomass_min", "initial_biomass_max"):
                # Per-project FG override (positive integers only); do not write to library.
                parsed = None
                if val not in (None, ""):
                    try:
                        parsed = int(val)
                        if parsed < 0:
                            parsed = None
                    except (TypeError, ValueError):
                        parsed = None
                if key == "initial_biomass_min":
                    initial_biomass_min_val = parsed
                else:
                    initial_biomass_max_val = parsed
                continue
            if isinstance(var, tk.BooleanVar):
                config[key] = val
            else:
                raw_val = val
                if raw_val in (None, ""):
                    # Empty field → treat as 0.0 (legacy behaviour for
                    # optional numeric fields).
                    config[key] = 0.0
                else:
                    try:
                        config[key] = float(raw_val)
                    except (TypeError, ValueError):
                        messagebox.showerror(
                            "Invalid value",
                            f"Field '{key}' is not a valid number: {raw_val!r}.\n\n"
                            "Use '.' as decimal separator (e.g. 1.5, not 1,5). "
                            "No changes were saved.",
                        )
                        return
                # Clamp movement_speed to a physical maximum of 1.0 cell/tick
                if key == "movement_speed":
                    if config[key] > 1.0:
                        config[key] = 1.0
                        var.set("1.0")
                # Clamp maintenance level (u_X) to [0, 1].
                if key == "maintenance_level":
                    if config[key] < 0.0:
                        config[key] = 0.0
                        var.set("0.0")
                    elif config[key] > 1.0:
                        config[key] = 1.0
                        var.set("1.0")
                # Clamp indivisible weight to [0, 10000] kg. 0 = continuous.
                if key == "min_split_biomass":
                    if config[key] < 0.0:
                        config[key] = 0.0
                        var.set("0.0")
                    elif config[key] > 10000.0:
                        config[key] = 10000.0
                        var.set("10000.0")

        # Validate biomass range: both must be set together, and max >= min.
        if (initial_biomass_min_val is None) != (initial_biomass_max_val is None):
            messagebox.showwarning(
                "Invalid Biomass Range",
                "Both Min and Max for 'Initial Total Biomass Range (ton)' must be set, or both left empty.",
            )
            return
        if (
            initial_biomass_min_val is not None
            and initial_biomass_max_val is not None
            and initial_biomass_max_val < initial_biomass_min_val
        ):
            messagebox.showwarning(
                "Invalid Biomass Range",
                "'Max' must be greater than or equal to 'Min' for the Initial Total Biomass Range.",
            )
            return

        # Strip initial_biomass* from library-bound config; it lives on the
        # project FG entry only.
        config.pop("initial_biomass", None)
        config.pop("initial_biomass_min", None)
        config.pop("initial_biomass_max", None)

        # Capture the spawn block from the active editor. Use strict mode
        # so unparseable spawn params / ref weights surface as an error
        # dialog instead of being silently dropped or coerced to 1.0.
        spawn_store = self.spawn_vars if is_dm else self.ndm_spawn_vars
        spawn_errors = []
        spawn_dict = self._collect_spawn_dict(spawn_store, errors=spawn_errors)
        if spawn_errors:
            messagebox.showerror(
                "Invalid spawn settings",
                "The spawn editor has invalid values:\n\n  - "
                + "\n  - ".join(spawn_errors)
                + "\n\nNo changes were saved.",
            )
            return
        # Mode='uniform' with no extra params is the default → keep it omitted
        # so legacy library entries stay clean. Anything else is persisted.
        if spawn_dict.get("mode") == "uniform" and len(spawn_dict) == 1:
            config.pop("spawn", None)
        else:
            config["spawn"] = spawn_dict

        # Biological sanity gate (per mareld_resume.txt Section 23, "Hard gate"):
        # for decision makers, the hypothetical per-tick energy balance at full
        # hunger from a single eat action must exceed the resting cost.
        #   intake_at_h1 = max_intake_rate * energy_gain   (best prey)
        #   feed_cost    = feeding_cost    * resting_metabolism
        #   rest_cost    = resting_cost(=1.0) * resting_metabolism
        #   netto_eat    = intake_at_h1 - feed_cost
        #   Hard gate:   netto_eat > rest_cost
        # If the hard gate is violated, abort the apply.
        #
        # NOTE on data-integrity: ``energy_gain`` is, by definition, the prey's
        # ``energy_content`` (MJ/ton). The matrix editor under "FG Interactions"
        # only exposes ``preys_on``; it never writes ``energy_gain``. To
        # prevent silent zero-intake when a hand-edited or legacy YAML row
        # lacks ``energy_gain``, we resolve intake using the prey's
        # ``energy_content`` from ``species_definitions`` as the authoritative
        # source. Any per-interaction ``energy_gain`` override is honoured if
        # present (back-compat), but missing values no longer silently
        # collapse to 0. ``max_intake_rate`` is now a per-predator property
        # read from the FG Editor (config[...]) rather than per-interaction.
        if is_dm:
            feeding_cost = float(config.get("feeding_cost", 0.0) or 0.0)
            resting_metabolism = float(config.get("resting_metabolism", 0.0) or 0.0)
            feed_cost = feeding_cost * resting_metabolism
            rest_cost = 1.0 * resting_metabolism  # resting_cost is hardcoded to 1.0
            try:
                mir = float(config.get("max_intake_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                mir = 0.0
            interactions = self.global_library.get("interaction_definitions", {}) or {}
            species_defs = self.global_library.get("species_definitions", {}) or {}
            best_intake = 0.0
            best_prey = None
            missing_energy = []  # prey_ids where energy_content is also missing
            prefix = f"{fg_id}_preys_on_"
            for key, entry in interactions.items():
                if not key.startswith(prefix):
                    continue
                if not isinstance(entry, dict) or not entry.get("preys_on"):
                    continue
                prey_id = key[len(prefix):]
                # Resolve energy gain: explicit override > prey energy_content.
                eg = None
                if "energy_gain" in entry and entry.get("energy_gain") not in (None, ""):
                    try:
                        eg = float(entry.get("energy_gain"))
                    except (TypeError, ValueError):
                        eg = None
                if eg is None:
                    prey_def = species_defs.get(prey_id, {}) or {}
                    ec = prey_def.get("energy_content")
                    if ec not in (None, ""):
                        try:
                            eg = float(ec)
                        except (TypeError, ValueError):
                            eg = None
                if eg is None:
                    eg = 0.0
                    if mir > 0.0:
                        missing_energy.append(prey_id)
                intake = mir * eg
                if intake > best_intake:
                    best_intake = intake
                    best_prey = prey_id
            if missing_energy:
                messagebox.showinfo(
                    "Missing Energy Data",
                    (
                        f"Cannot evaluate energy balance for '{fg_id}': the "
                        f"following prey species have neither an explicit "
                        f"'energy_gain' on the interaction nor an "
                        f"'energy_content' in species_definitions:\n  "
                        + ", ".join(missing_energy)
                        + "\n\nLibrary and project entry have NOT been "
                        "updated. Set 'energy_content' on each prey FG "
                        "(FG Editor → Energy Content) before applying."
                    ),
                )
                return
            netto_eat = best_intake - feed_cost
            eat_minus_rest = netto_eat - rest_cost
            if eat_minus_rest <= 0.0:
                prey_txt = best_prey if best_prey else "(no prey with preys_on: true found)"
                messagebox.showinfo(
                    "Invalid Energy Balance",
                    (
                        f"Changes not accepted for '{fg_id}'.\n\n"
                        f"The hypothetical per-tick energy balance fails the "
                        f"hard gate (netto_eat must exceed rest_cost):\n"
                        f"  intake = max_intake_rate * energy_gain"
                        f" = {best_intake:.3f}\n"
                        f"  feed_cost = feeding_cost * resting_metabolism"
                        f" = {feed_cost:.3f}\n"
                        f"  rest_cost = resting_cost(1.0) * resting_metabolism"
                        f" = {rest_cost:.3f}\n"
                        f"  netto_eat = intake - feed_cost = {netto_eat:.3f}\n"
                        f"  netto_eat - rest_cost = {eat_minus_rest:.3f}"
                        f"  (must be > 0)\n\n"
                        f"Best prey considered: {prey_txt}.\n"
                        "Library and project entry have NOT been updated. "
                        "Adjust feeding_cost, resting_metabolism, or the "
                        "predation max_intake_rate / energy_gain so that "
                        "netto_eat > rest_cost."
                    ),
                )
                return

        self.current_fg_configs[fg_id] = config

        # Persist initial_biomass range on the project FG entry (per-project value).
        # Always remove the legacy scalar key so projects converge on the new schema.
        fg_entry = fgs[idx]
        fg_entry.pop("initial_biomass", None)
        if initial_biomass_min_val is None or initial_biomass_max_val is None:
            fg_entry.pop("initial_biomass_min", None)
            fg_entry.pop("initial_biomass_max", None)
        else:
            fg_entry["initial_biomass_min"] = int(initial_biomass_min_val)
            fg_entry["initial_biomass_max"] = int(initial_biomass_max_val)

        # Sync remaining fields with global library
        if "species_definitions" not in self.global_library:
            self.global_library["species_definitions"] = {}
        self.global_library["species_definitions"][fg_id] = config

        # Invariant: for every interaction ``<predator>_preys_on_<fg_id>``,
        # ``energy_gain`` MUST equal this FG's ``energy_content`` (MJ/ton).
        # The matrix editor never writes ``energy_gain`` (it's a legacy field
        # only read by the hard-gate validator with fallback to
        # ``prey.energy_content``). If we don't sync it here, raising
        # ``energy_content`` in the FG Editor leaves stale ``energy_gain``
        # values on every predator relation — exactly the drift that
        # produced the 6/8 mismatches we cleaned up earlier.
        synced_refs = []
        new_ec = config.get("energy_content")
        if new_ec not in (None, ""):
            try:
                new_ec_f = float(new_ec)
            except (TypeError, ValueError):
                new_ec_f = None
            if new_ec_f is not None:
                interactions = self.global_library.setdefault(
                    "interaction_definitions", {})
                suffix = f"_preys_on_{fg_id}"
                for ikey, entry in interactions.items():
                    if not ikey.endswith(suffix):
                        continue
                    if not isinstance(entry, dict):
                        continue
                    if not entry.get("preys_on"):
                        continue
                    old = entry.get("energy_gain")
                    try:
                        old_f = float(old) if old not in (None, "") else None
                    except (TypeError, ValueError):
                        old_f = None
                    if old_f != new_ec_f:
                        entry["energy_gain"] = new_ec_f
                        synced_refs.append((ikey, old, new_ec_f))

        self.save_yaml(self.global_library, self.library_path)
        # FG editor writes initial_biomass min/max and spawn block onto the
        # project FG entry — these are NOT persisted by the library save
        # above and require Save Project.
        self._mark_dirty()
        if synced_refs:
            details = "\n".join(
                f"  - {k}.energy_gain: {old!r} → {new}"
                for k, old, new in synced_refs
            )
            messagebox.showinfo(
                "Success",
                f"Updated {fg_id}. Library updated; initial biomass range "
                f"saved on project entry (remember to Save Project).\n\n"
                f"Auto-synced energy_gain on {len(synced_refs)} predation "
                f"relation(s) to match this FG's energy_content="
                f"{new_ec_f}:\n{details}",
            )
        else:
            messagebox.showinfo(
                "Success",
                f"Updated {fg_id}. Library updated; initial biomass range "
                f"saved on project entry (remember to Save Project).",
            )

    def apply_matrix_changes(self):
        if "interaction_definitions" not in self.global_library:
            self.global_library["interaction_definitions"] = {}
            
        for key, data in self.matrix_entries.items():
            if key not in self.global_library["interaction_definitions"]:
                self.global_library["interaction_definitions"][key] = {}
            
            for data_key, var in data.items():
                if isinstance(var, tk.BooleanVar):
                    self.global_library["interaction_definitions"][key][data_key] = bool(var.get())
                else:
                    val_str = var.get()
                    if val_str:
                        try:
                            val = float(val_str)
                            self.global_library["interaction_definitions"][key][data_key] = val
                        except ValueError:
                            pass # skip invalid

        # Persist impact tables (list of {value, biomass_factor, energy_factor})
        for key, table in getattr(self, "matrix_tables", {}).items():
            if key not in self.global_library["interaction_definitions"]:
                self.global_library["interaction_definitions"][key] = {}
            self.global_library["interaction_definitions"][key]["impact_table"] = [
                {
                    "value": float(row.get("value", 0.0)),
                    "biomass_factor": float(row.get("biomass_factor", 0.0)),
                    "energy_factor": float(row.get("energy_factor", 0.0)),
                }
                for row in table
            ]

        self.save_yaml(self.global_library, self.library_path)
        messagebox.showinfo("Success", "Updated interactions and saved to library.")

    def open_impact_table_editor(self, key, fg_id, impact_id):
        """Open a dialog to edit the impact table (value, biomass_factor, energy_factor) for (FG, impact).

        All three columns are in [0, 1]. Linear interpolation is used at lookup, clipped to
        last value at the boundaries (no extrapolation).
        """
        if not hasattr(self, "matrix_tables"):
            self.matrix_tables = {}
        current = [dict(r) for r in self.matrix_tables.get(key, [])]

        impact_name = self.global_library.get("impact_definitions", {}).get(impact_id, {}).get("display_name", impact_id)
        fg_name = self.fg_display(fg_id)

        # Look up the physical value range for this impact, set in the Impact
        # Editor (Project & FGs tab) as value_min/value_max per project.
        impact_entry = self._find_impact_entry(impact_id)
        def _coerce_f(v):
            try:
                if v is None or v == "":
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None
        v_min = _coerce_f(impact_entry.get('value_min')) if impact_entry else None
        v_max = _coerce_f(impact_entry.get('value_max')) if impact_entry else None
        has_range = v_min is not None and v_max is not None and v_max >= v_min

        top = tk.Toplevel(self.root)
        top.title(f"Impact Table — {fg_name} × {impact_name}")
        top.transient(self.root)

        unit = self._impact_unit(impact_id)
        if has_range:
            info_text = (
                f"'Value' must be in [{v_min:g}, {v_max:g}] {unit} "
                f"(range set in the Impact Editor as Min/Max for this impact). "
                "'Biomass factor' and 'Energy factor' must be in [0, 1]. "
                "Linear interpolation between rows; outside the range, "
                "the nearest endpoint value is used (no extrapolation)."
            )
        else:
            info_text = (
                "'Value' range is not set for this impact — open the Impact "
                "Editor (Project & FGs tab) and set Min/Max to enable live "
                "range validation here. "
                "'Biomass factor' and 'Energy factor' must be in [0, 1]. "
                "Linear interpolation between rows; outside the range, "
                "the nearest endpoint value is used (no extrapolation)."
            )
        info = ttk.Label(top, text=info_text, wraplength=480, justify="left")
        info.pack(padx=10, pady=(10, 5), anchor="w")

        table_frame = ttk.Frame(top)
        table_frame.pack(padx=10, pady=5, fill="both", expand=True)

        headers = (f"Value ({self._impact_unit(impact_id)})", "Biomass factor", "Energy factor")
        for j, h in enumerate(headers):
            ttk.Label(table_frame, text=h, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=j, padx=4, pady=2)

        row_vars = []  # list of (value_var, bf_var, ef_var)

        # Live validators: 'value' accepts any numeric (or partial) string; the
        # factor columns reject any intermediate string that cannot be the prefix
        # of a number in [0, 1] (so values outside the unit interval cannot even
        # be typed).
        def _validate_value(proposed):
            # Allow empty / partial inputs that could still complete to a valid
            # number; when a numeric range is set on the impact, also forbid
            # typing anything outside [v_min, v_max].
            if proposed in ("", "-", "+", ".", "-.", "+."):
                return True
            try:
                v = float(proposed)
            except ValueError:
                return False
            if has_range:
                return v_min <= v <= v_max
            return True

        def _validate_unit(proposed):
            # Allow empty / partial inputs that could still become a valid value in [0, 1].
            if proposed in ("", "."):
                return True
            try:
                v = float(proposed)
            except ValueError:
                return False
            return 0.0 <= v <= 1.0

        vcmd_value = (top.register(_validate_value), "%P")
        vcmd_unit = (top.register(_validate_unit), "%P")

        def add_row(value=0.0, bf=0.0, ef=0.0):
            r = len(row_vars) + 1
            v_var = tk.StringVar(value=f"{float(value):.4g}")
            b_var = tk.StringVar(value=f"{float(bf):.4g}")
            e_var = tk.StringVar(value=f"{float(ef):.4g}")
            ttk.Entry(table_frame, textvariable=v_var, width=10,
                      validate="key", validatecommand=vcmd_value).grid(row=r, column=0, padx=4, pady=2)
            ttk.Entry(table_frame, textvariable=b_var, width=10,
                      validate="key", validatecommand=vcmd_unit).grid(row=r, column=1, padx=4, pady=2)
            ttk.Entry(table_frame, textvariable=e_var, width=10,
                      validate="key", validatecommand=vcmd_unit).grid(row=r, column=2, padx=4, pady=2)
            row_vars.append((v_var, b_var, e_var))

        for row in current:
            add_row(row.get("value", 0.0), row.get("biomass_factor", 0.0), row.get("energy_factor", 0.0))
        if not row_vars:
            add_row()

        btns = ttk.Frame(top)
        btns.pack(padx=10, pady=(2, 10), fill="x")

        def on_add_row():
            add_row()

        def on_remove_last():
            if not row_vars:
                return
            v, b, e = row_vars.pop()
            # Remove the corresponding entry widgets (last row in the grid)
            r = len(row_vars) + 1
            for col in range(3):
                w = table_frame.grid_slaves(row=r, column=col)
                for wd in w:
                    wd.destroy()

        def _parse_float(s):
            try:
                return float(s)
            except (TypeError, ValueError):
                return None

        def _parse_unit(s):
            v = _parse_float(s)
            if v is None or v < 0.0 or v > 1.0:
                return None
            return v

        def on_save():
            new_table = []
            for v_var, b_var, e_var in row_vars:
                v = _parse_float(v_var.get())
                b = _parse_unit(b_var.get())
                e = _parse_unit(e_var.get())
                if v is None:
                    messagebox.showwarning(
                        "Invalid value",
                        "'Value' cells must be numeric.", parent=top,
                    )
                    return
                if has_range and not (v_min <= v <= v_max):
                    messagebox.showwarning(
                        "Out of range",
                        f"'Value' cells must be within [{v_min:g}, {v_max:g}] "
                        f"for this impact.", parent=top,
                    )
                    return
                if b is None or e is None:
                    messagebox.showwarning(
                        "Invalid value",
                        "'Biomass factor' and 'Energy factor' cells must be numeric values within [0, 1].",
                        parent=top,
                    )
                    return
                new_table.append({"value": v, "biomass_factor": b, "energy_factor": e})
            # Sort by value for predictable interpolation
            new_table.sort(key=lambda r: r["value"])
            self.matrix_tables[key] = new_table
            top.destroy()

        ttk.Button(btns, text="Add Row", command=on_add_row).pack(side="left")
        ttk.Button(btns, text="Remove Last", command=on_remove_last).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right")
        ttk.Button(btns, text="Save", command=on_save).pack(side="right", padx=5)

    def add_from_library(self, category="decision_makers"):
        all_lib_fgs = list(self.global_library.get("species_definitions", {}).keys())
        # Filter what's selectable based on category:
        # - Non Decision Makers: only phytoplankton may be added.
        # - Decision Makers: phytoplankton is not selectable.
        if category == "non_decision_makers":
            lib_fgs = [fg for fg in all_lib_fgs if fg == "phytoplankton"]
        else:
            lib_fgs = [fg for fg in all_lib_fgs if fg != "phytoplankton"]
        # Hide groups that are already part of the project (in any category).
        existing_ids = set(self._all_fg_ids())
        lib_fgs = [fg for fg in lib_fgs if fg not in existing_ids]
        if not lib_fgs:
            messagebox.showinfo("Library Empty", "No selectable groups available for this category.")
            return

        top = tk.Toplevel(self.root)
        top.title("Select from Library")
        top.transient(self.root)

        ttk.Label(top, text="Select one or more groups (Ctrl/Shift-click for multi-select):").pack(padx=10, pady=(10, 2), anchor="w")

        list_frame = ttk.Frame(top)
        list_frame.pack(padx=10, pady=5, fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        lb = tk.Listbox(list_frame, selectmode="extended", height=min(15, max(5, len(lib_fgs))),
                        width=30, exportselection=False, yscrollcommand=scrollbar.set)
        scrollbar.config(command=lb.yview)
        lb.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        lib_fgs.sort(key=lambda fg: self.fg_display(fg).lower())
        for item in lib_fgs:
            lb.insert("end", self.fg_display(item))

        btn_frame = ttk.Frame(top)
        btn_frame.pack(padx=10, pady=(2, 10), fill="x")

        def select_all():
            lb.selection_set(0, "end")

        def do_add():
            selection = lb.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select at least one group.", parent=top)
                return
            added = 0
            for i in selection:
                fg_id = lib_fgs[i]
                if fg_id in self._all_fg_ids():
                    continue
                new_entry = {'group_id': fg_id}
                # Seed initial biomass range from library defaults (supports both
                # the new min/max schema and the legacy scalar `initial_biomass`).
                lib_mn, lib_mx = self._read_initial_biomass_range({}, fg_id)
                if lib_mn is not None and lib_mx is not None:
                    new_entry['initial_biomass_min'] = int(lib_mn)
                    new_entry['initial_biomass_max'] = int(lib_mx)
                self.project_data.setdefault(category, []).append(new_entry)
                self.current_fg_configs[fg_id] = self.global_library["species_definitions"][fg_id]
                # Keep the library's is_decision_maker flag in sync with the
                # category the user chose to add it to.
                lib_entry = self.global_library["species_definitions"].get(fg_id, {})
                desired_dm = (category == "decision_makers")
                if lib_entry.get("is_decision_maker") != desired_dm:
                    lib_entry["is_decision_maker"] = desired_dm
                    self.global_library["species_definitions"][fg_id] = lib_entry
                    self.save_yaml(self.global_library, self.library_path)
                added += 1
            self.update_fg_list()
            self.refresh_matrix()
            if added:
                self._mark_dirty()
            top.destroy()

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(side="left")
        ttk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side="right")
        ttk.Button(btn_frame, text="Add", command=do_add).pack(side="right", padx=5)
        lb.bind("<Double-Button-1>", lambda e: do_add())

    def add_new_fg(self, category="decision_makers"):
        import tkinter.simpledialog as sd
        fg_id = sd.askstring("New FG", "Enter ID for new Functional Group (English):")
        if fg_id:
            if fg_id in self._all_fg_ids():
                messagebox.showerror("Error", "FG ID already exists in project.")
                return

            is_dm = (category == "decision_makers")
            self.project_data.setdefault(category, []).append(
                {'group_id': fg_id, 'initial_biomass_min': 0, 'initial_biomass_max': 0}
            )
            self.current_fg_configs[fg_id] = {
                "display_name": fg_id,
                "is_decision_maker": is_dm,
                "growth_rate": 0.0,
                "max_energy_reserve": 0.0,
                "resting_metabolism": 0.0,
                "maintenance_level": 0.0,
                "movement_speed": 0.0,
                "movement_cost": 3.0,
                "feeding_cost": 3.0,
            }
            # Add to global library immediately
            if "species_definitions" not in self.global_library:
                self.global_library["species_definitions"] = {}
            self.global_library["species_definitions"][fg_id] = self.current_fg_configs[fg_id]
            self.save_yaml(self.global_library, self.library_path)
            
            self.update_fg_list()
            self.refresh_matrix()
            self._mark_dirty()

    def remove_fg(self, category="decision_makers"):
        listbox = self._listbox_for(category)
        selection = listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        fgs = self.project_data.get(category, [])
        if idx >= len(fgs):
            return
        fg_id = fgs[idx]['group_id']
        self.project_data[category] = [fg for fg in fgs if fg['group_id'] != fg_id]
        if fg_id in self.current_fg_configs:
            del self.current_fg_configs[fg_id]
        if self.active_fg_category == category:
            self.active_fg_category = None
        self.update_fg_list()
        self.refresh_matrix()
        self._mark_dirty()

    def fg_display(self, fg_id, include_sv=False):
        """Return display label for a functional group with capitalized first letter."""
        cfg = self.current_fg_configs.get(fg_id, {})
        name = cfg.get("display_name") if isinstance(cfg, dict) else None
        if not name:
            name = self.global_library.get("species_definitions", {}).get(fg_id, {}).get("display_name", fg_id)
        if not name:
            name = fg_id
        
        display = name[:1].upper() + name[1:] if name else name
        
        if include_sv:
            sv_name = self.sv_mapping.get(fg_id)
            if sv_name:
                display = f"{display}/{sv_name}"
        return display

    def update_fg_list(self):
        # Sort underlying project lists alphabetically so listbox indices map
        # directly to project_data entries.
        if self.project_data.get('decision_makers'):
            self.project_data['decision_makers'].sort(
                key=lambda fg: self.fg_display(fg['group_id']).lower())
        if self.project_data.get('non_decision_makers'):
            self.project_data['non_decision_makers'].sort(
                key=lambda fg: self.fg_display(fg['group_id']).lower())
        self.fg_listbox.delete(0, "end")
        for fg in self.project_data.get('decision_makers', []) or []:
            self.fg_listbox.insert("end", self.fg_display(fg['group_id']))
        if hasattr(self, 'ndm_listbox'):
            self.ndm_listbox.delete(0, "end")
            for fg in self.project_data.get('non_decision_makers', []) or []:
                self.ndm_listbox.insert("end", self.fg_display(fg['group_id']))
        # Mirror the project FG list into the Inference tab so its inputs
        # always reflect the current set of project FGs.
        if hasattr(self, 'refresh_inference_tab'):
            self.refresh_inference_tab()
        # Apply muted styling and refresh mute-button labels.
        self._apply_listbox_mute_styling()
        self._refresh_mute_button_labels()

    def update_impact_list(self):
        def _impact_display(iv):
            impact_id = iv['impact_id']
            return self.global_library.get("impact_definitions", {}).get(impact_id, {}).get("display_name", impact_id)
        def _impact_display_with_unit(iv):
            base = f"{_impact_display(iv)} ({self._impact_unit(iv['impact_id'])})"
            # Append [observable] marker so the project list makes it obvious
            # which impacts feed into the policy network's input layer.
            if iv.get('observable'):
                base = f"{base} [observable]"
            return base
        if self.project_data.get('impact_variables'):
            self.project_data['impact_variables'].sort(key=lambda iv: _impact_display(iv).lower())
        self.impact_listbox.delete(0, "end")
        for iv in self.project_data.get('impact_variables', []):
            self.impact_listbox.insert("end", _impact_display_with_unit(iv))
        self._apply_listbox_mute_styling()
        self._refresh_mute_button_labels()
        if hasattr(self, 'impact_editor_frame'):
            self.on_impact_select()
        if hasattr(self, 'inference_impact_frame'):
            self._refresh_inference_impact_maps()

    def add_impact_from_library(self):
        lib_impacts = list(self.global_library.get("impact_definitions", {}).keys())
        # Hide impacts that are already part of the project.
        existing_ids = {iv['impact_id'] for iv in self.project_data.get('impact_variables', []) or []}
        lib_impacts = [imp for imp in lib_impacts if imp not in existing_ids]
        if not lib_impacts:
            messagebox.showinfo("Library Empty", "No selectable impact variables available.")
            return

        top = tk.Toplevel(self.root)
        top.title("Select Impact Variables from Library")
        top.transient(self.root)

        ttk.Label(top, text="Select one or more impacts (Ctrl/Shift-click for multi-select):").pack(padx=10, pady=(10, 2), anchor="w")

        list_frame = ttk.Frame(top)
        list_frame.pack(padx=10, pady=5, fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        lb = tk.Listbox(list_frame, selectmode="extended", height=min(15, max(5, len(lib_impacts))),
                        width=30, exportselection=False, yscrollcommand=scrollbar.set)
        scrollbar.config(command=lb.yview)
        lb.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        lib_impacts.sort(key=lambda imp: self.global_library["impact_definitions"][imp].get("display_name", imp).lower())
        for item in lib_impacts:
            display = self.global_library["impact_definitions"][item].get("display_name", item)
            lb.insert("end", display)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(padx=10, pady=(2, 10), fill="x")

        def select_all():
            lb.selection_set(0, "end")

        def do_add():
            selection = lb.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select at least one impact.", parent=top)
                return
            for i in selection:
                impact_id = lib_impacts[i]
                if not any(iv['impact_id'] == impact_id for iv in self.project_data.get('impact_variables', [])):
                    self.project_data.setdefault('impact_variables', []).append({'impact_id': impact_id})
            self.update_impact_list()
            self.refresh_matrix()
            self._mark_dirty()
            top.destroy()

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(side="left")
        ttk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side="right")
        ttk.Button(btn_frame, text="Add", command=do_add).pack(side="right", padx=5)
        lb.bind("<Double-Button-1>", lambda e: do_add())

    def remove_impact(self):
        selection = self.impact_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        impact_vars = self.project_data.get('impact_variables', [])
        if idx >= len(impact_vars):
            return
        impact_id = impact_vars[idx]['impact_id']
        self.project_data['impact_variables'] = [iv for iv in impact_vars if iv['impact_id'] != impact_id]
        self.update_impact_list()
        self.refresh_matrix()
        self._mark_dirty()

    def new_project(self):
        self.project_data = {
            "project_metadata": {
                "name": "New Project",
                "reference_grid_width": 60,
                "reference_grid_height": 60,
            },
            "simulation_settings": {},
            "decision_makers": [],
            "non_decision_makers": [],
            "impact_variables": []
        }
        self.current_fg_configs = {}
        self.active_fg_category = None
        self.project_name_var.set("New Project")
        if hasattr(self, 'ref_grid_w_var'):
            self.ref_grid_w_var.set("60")
            self.ref_grid_h_var.set("60")
        self.update_fg_list()
        self.update_impact_list()
        self.refresh_matrix()
        # Fresh project starts clean (the project_name_var trace fires
        # during set() above and would otherwise mark dirty).
        self._clear_dirty()

    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if path:
            self.load_project_from_path(path)

    def load_project_from_path(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Not Found", f"Project file no longer exists:\n{path}")
            self.recent_projects = [p for p in self.recent_projects if p != path]
            self.save_recent_projects()
            self.refresh_recent_menu()
            return
        data = self.load_yaml(path)
        if data:
            self.project_data = data
            self.project_path = path
            self.project_name_var.set(data.get("project_metadata", {}).get("name", "Unnamed Project"))
            # Load reference grid (default 60x60 for legacy projects that
            # predate this field).
            pmeta = data.get("project_metadata", {}) or {}
            # Clamp to MIN_REF (3) — values below the live-validation floor
            # in legacy / hand-edited project files are rounded up so the
            # runtime never sees a sub-3 reference grid.
            _MIN_REF = getattr(self, 'ref_grid_min', 3)
            def _coerce_pos_int(v, default):
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    return default
                if iv < _MIN_REF:
                    return _MIN_REF
                return iv
            rw = _coerce_pos_int(pmeta.get("reference_grid_width"), 60)
            rh = _coerce_pos_int(pmeta.get("reference_grid_height"), 60)
            if hasattr(self, 'ref_grid_w_var'):
                self.ref_grid_w_var.set(str(rw))
                self.ref_grid_h_var.set(str(rh))
            # Backward compatibility: legacy projects had a single `functional_groups` list.
            # Split it into decision/non-decision based on the library's is_decision_maker flag.
            if 'functional_groups' in self.project_data and (
                'decision_makers' not in self.project_data
                and 'non_decision_makers' not in self.project_data
            ):
                dms, ndms = [], []
                for fg in self.project_data.get('functional_groups', []) or []:
                    gid = fg.get('group_id')
                    if not gid:
                        continue
                    lib_entry = self.global_library.get("species_definitions", {}).get(gid, {})
                    if lib_entry.get('is_decision_maker', False):
                        dms.append({'group_id': gid})
                    else:
                        ndms.append({'group_id': gid})
                self.project_data['decision_makers'] = dms
                self.project_data['non_decision_makers'] = ndms
                self.project_data.pop('functional_groups', None)
            self.project_data.setdefault('decision_makers', [])
            self.project_data.setdefault('non_decision_makers', [])
            self.active_fg_category = None
            # Load configs for active FGs
            self.current_fg_configs = {}
            for gid in self._all_fg_ids():
                if gid in self.global_library.get("species_definitions", {}):
                    self.current_fg_configs[gid] = self.global_library["species_definitions"][gid]
                else:
                    self.current_fg_configs[gid] = {"display_name": gid}
            if 'impact_variables' not in self.project_data:
                self.project_data['impact_variables'] = []
            self.update_fg_list()
            self.update_impact_list()
            self.refresh_matrix()
            self.add_to_recent(path)
            # Loading from disk = clean state. Done after all .set()
            # calls above so the Tk-variable traces don't leave the
            # project marked dirty.
            self._clear_dirty()

    def save_project(self):
        """Persist project_data to YAML. Returns True on success, False if
        the user cancelled the path dialog or the save did not happen."""
        if not self.project_path:
            self.project_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if self.project_path:
            self.project_data["project_metadata"]["name"] = self.project_name_var.get()
            # Persist reference grid. Empty / invalid -> 60. Positive but
            # below MIN_REF (3) -> clamped to MIN_REF so the saved file is
            # always consistent with the live-validation rule.
            _MIN_REF = getattr(self, 'ref_grid_min', 3)
            def _coerce_pos_int(v, default):
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    return default
                if iv < _MIN_REF:
                    return _MIN_REF
                return iv
            rw = _coerce_pos_int(self.ref_grid_w_var.get(), 60)
            rh = _coerce_pos_int(self.ref_grid_h_var.get(), 60)
            self.project_data["project_metadata"]["reference_grid_width"] = rw
            self.project_data["project_metadata"]["reference_grid_height"] = rh
            # Reflect the clamped value back into the UI so the user sees what
            # was actually saved.
            self.ref_grid_w_var.set(str(rw))
            self.ref_grid_h_var.set(str(rh))
            self.save_yaml(self.project_data, self.project_path)
            self.add_to_recent(self.project_path)
            self._clear_dirty()
            messagebox.showinfo("Success", f"Project saved to {self.project_path}")
            return True
        return False

    def load_recent_projects(self):
        if not os.path.exists(self.recent_path):
            return []
        try:
            with open(self.recent_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return lines[:self.max_recent]
        except Exception:
            return []

    def save_recent_projects(self):
        try:
            with open(self.recent_path, 'w', encoding='utf-8') as f:
                for p in self.recent_projects[:self.max_recent]:
                    f.write(p + "\n")
        except Exception as e:
            print(f"Could not save recent projects: {e}")

    def add_to_recent(self, path):
        if not path:
            return
        path = os.path.abspath(path)
        self.recent_projects = [p for p in self.recent_projects if p != path]
        self.recent_projects.insert(0, path)
        self.recent_projects = self.recent_projects[:self.max_recent]
        self.save_recent_projects()
        self.refresh_recent_menu()

    def refresh_recent_menu(self):
        if not hasattr(self, 'recent_menu'):
            return
        self.recent_menu.delete(0, "end")
        if not self.recent_projects:
            self.recent_menu.add_command(label="(No recent projects)", state="disabled")
            return
        for i, path in enumerate(self.recent_projects, start=1):
            label = f"{i}. {os.path.basename(path)}  \u2014  {path}"
            self.recent_menu.add_command(label=label, command=lambda p=path: self.load_project_from_path(p))
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Clear Recent", command=self.clear_recent)

    def clear_recent(self):
        self.recent_projects = []
        self.save_recent_projects()
        self.refresh_recent_menu()

if __name__ == "__main__":
    # Use a DnD-enabled root window when ``tkinterdnd2`` is installed so that
    # the Inference-tab impact-map drop zones accept dragged files. Falls back
    # to a regular ``tk.Tk()`` when the package is missing (drop zones then
    # behave as click-to-browse only).
    root = None
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = None
    if root is None:
        root = tk.Tk()
    app = FGConfigApp(root)
    # Auto-load the most recently used project, if any still exists on disk.
    try:
        for _recent in list(app.recent_projects):
            if os.path.exists(_recent):
                app.load_project_from_path(_recent)
                break
    except Exception as _e:
        print(f"Could not auto-load recent project: {_e}")
    root.mainloop()
