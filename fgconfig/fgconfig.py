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
        self.root.title("Ecosystem FG Configuration Tool")
        self.root.geometry("1000x700")

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
        filemenu.add_command(label="Exit", command=self.root.quit)
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

        # Tab 2: FG Interactions
        self.matrix_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_tab, text="FG Interactions")

        # Tab 3: Impact Interactions
        self.impact_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.impact_tab, text="Impact Interactions")

        self.setup_matrix_tab()

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
        
        ttk.Label(info_frame, text="Project Name:").grid(row=0, column=0, sticky="w", padx=5)
        self.project_name_var = tk.StringVar(value="New Project")
        ttk.Entry(info_frame, textvariable=self.project_name_var).grid(row=0, column=1, sticky="ew", padx=5)

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

        # Track which list the FG editor is currently bound to
        self.active_fg_category = None

        # Impact Variables list in project
        impact_frame = ttk.LabelFrame(self.project_inner, text="Impact Variables in Project")
        impact_frame.pack(expand=True, fill="both", padx=10, pady=5)

        self.impact_listbox = tk.Listbox(impact_frame)
        self.impact_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        impact_btn_frame = ttk.Frame(impact_frame)
        impact_btn_frame.pack(side="right", fill="y", padx=5, pady=5)

        ttk.Button(impact_btn_frame, text="Add from Library", command=self.add_impact_from_library).pack(fill="x", pady=2)
        ttk.Button(impact_btn_frame, text="Remove Impact", command=self.remove_impact).pack(fill="x", pady=2)

        # FG Editor for Decision Makers
        self.editor_frame = ttk.LabelFrame(self.project_inner, text="FG Editor")
        # Note: not packed here; on_fg_select shows/hides editors based on category.

        self.prop_vars = {}
        main_props = [
            ("Max Energy Reserve (ME_X MJ/ton)", "max_energy_reserve", "entry"),
            ("Energy Content (MJ/ton)", "energy_content", "entry"),
            ("Resting Metabolism (MJ/ton)", "resting_metabolism", "entry"),
            ("Maintenance Level (u_X, fraction)", "maintenance_level", "entry"),
            ("Movement Speed (cells/tick)", "movement_speed", "entry"),
            ("Indivisible Weight (kg)", "min_split_biomass", "entry"),
            ("Initial Total Biomass (ton)", "initial_biomass", "entry")
        ]

        for i, (label, key, type) in enumerate(main_props):
            ttk.Label(self.editor_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            if type == "entry":
                var = tk.StringVar()
                ent = ttk.Entry(self.editor_frame, textvariable=var)
                ent.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                self.prop_vars[key] = var
            elif type == "check":
                var = tk.BooleanVar()
                chk = ttk.Checkbutton(self.editor_frame, variable=var)
                chk.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                self.prop_vars[key] = var

        # Action Costs on one row
        row_idx = len(main_props)
        ttk.Label(self.editor_frame, text="Action Costs").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)

        costs_frame = ttk.Frame(self.editor_frame)
        costs_frame.grid(row=row_idx, column=1, sticky="w", padx=5, pady=2)

        action_costs = [
            ("Eat:", "feeding_cost"),
            ("Rest:", "resting_cost"),
            ("Move:", "movement_cost")
        ]

        for j, (label, key) in enumerate(action_costs):
            ttk.Label(costs_frame, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            ent = ttk.Entry(costs_frame, textvariable=var, width=8)
            ent.pack(side="left", padx=(0, 10))
            self.prop_vars[key] = var

        ttk.Button(self.editor_frame, text="Apply Changes", command=self.apply_fg_changes).grid(row=row_idx + 1, column=0, columnspan=2, pady=5)

        # FG Editor for Non Decision Makers
        self.ndm_editor_frame = ttk.LabelFrame(self.project_inner, text="FG Editor")
        # Note: not packed here; on_fg_select shows/hides editors based on category.

        self.ndm_prop_vars = {}
        ndm_props = [
            ("Max Growth (fraction/tick)", "growth_rate"),
            ("Max Carrying Capacity (ton/cell)", "max_carrying_capacity"),
            ("Initial Total Biomass (ton)", "initial_biomass"),
        ]
        for i, (label, key) in enumerate(ndm_props):
            ttk.Label(self.ndm_editor_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar()
            ent = ttk.Entry(self.ndm_editor_frame, textvariable=var)
            ent.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.ndm_prop_vars[key] = var

        ttk.Button(self.ndm_editor_frame, text="Apply Changes", command=self.apply_fg_changes).grid(
            row=len(ndm_props), column=0, columnspan=2, pady=5
        )

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

    def refresh_matrix(self):
        for widget in self.matrix_container.winfo_children():
            widget.destroy()
        for widget in self.impact_container.winfo_children():
            widget.destroy()

        fgs = sorted(self._all_fg_ids(), key=lambda fg_id: self.fg_display(fg_id).lower())
        if not fgs:
            ttk.Label(self.matrix_container, text="Add FGs to the project to see interaction matrices.").pack(padx=10, pady=10)
            ttk.Label(self.impact_container, text="Add FGs to the project to see impact interactions.").pack(padx=10, pady=10)
            return

        self.matrix_entries = {}
        self.matrix_widgets = {}
        self.matrix_tables = {}
        
        # FG Interactions tab: Predation + Max Intake
        self.create_matrix_section("Predation (row eats column)", "preys_on", fgs, fgs, cell_type="bool",
                                   parent=self.matrix_container)

        self.create_matrix_section("Max Intake Rate (I_XY) [ton prey / ton consumer]", "max_intake_rate", fgs, fgs,
                                   parent=self.matrix_container)
        
        # Impact Interactions tab: Impact Affects (boolean) and Impact Tables (table editor per cell)
        impacts = [iv['impact_id'] for iv in self.project_data.get('impact_variables', [])]
        impacts.sort(key=lambda imp: self.global_library.get("impact_definitions", {}).get(imp, {}).get("display_name", imp).lower())
        if impacts:
            self.create_matrix_section("Impact Affects (row impacted by column)", "impact_affects", fgs, impacts,
                                       cell_type="bool", parent=self.impact_container)
            self.create_matrix_section("Impact Tables (value, biomass factor, energy factor)", "impact_table", fgs, impacts,
                                       cell_type="table", parent=self.impact_container)
        else:
            ttk.Label(self.impact_container, text="Add impact variables to the project to see impact interactions.").pack(padx=10, pady=10)

        # Link predation checkboxes to max_intake_rate entry enable-state
        for key, data in self.matrix_entries.items():
            preys_var = data.get("preys_on")
            if preys_var is None:
                continue
            intake_widget = self.matrix_widgets.get(key, {}).get("max_intake_rate")
            if intake_widget is None:
                continue
            def make_updater(var=preys_var, widget=intake_widget, k=key):
                def update(*_):
                    if var.get():
                        widget.configure(state="normal")
                    else:
                        # Clear value and disable
                        self.matrix_entries[k]["max_intake_rate"].set("")
                        widget.configure(state="disabled")
                return update
            updater = make_updater()
            preys_var.trace_add("write", updater)
            updater()

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

        ttk.Button(self.matrix_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)
        if impacts:
            ttk.Button(self.impact_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)

    def create_matrix_section(self, title, data_key, row_ids, col_ids, cell_type="entry", parent=None):
        if parent is None:
            parent = self.matrix_container
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="x", padx=10, pady=10)

        impact_keys = ("impact_affects", "impact_table")
        # Headers
        ttk.Label(frame, text="Group \\ Var").grid(row=0, column=0, padx=5, pady=5)
        for j, col_id in enumerate(col_ids):
            # Use display_name from impact_definitions if available
            if data_key in impact_keys:
                label_text = self.global_library.get("impact_definitions", {}).get(col_id, {}).get("display_name", col_id)
            else:
                label_text = self.fg_display(col_id)
            ttk.Label(frame, text=label_text).grid(row=0, column=j+1, padx=5, pady=5)

        for i, row_id in enumerate(row_ids):
            row_label = self.fg_display(row_id)
            ttk.Label(frame, text=row_label).grid(row=i+1, column=0, padx=5, pady=5)
            for j, col_id in enumerate(col_ids):
                # Unique key for storage
                if data_key in impact_keys:
                    key = f"{row_id}_impacted_by_{col_id}"
                else:
                    key = f"{row_id}_preys_on_{col_id}"
                
                if key not in self.matrix_entries:
                    self.matrix_entries[key] = {}

                # Look for existing value in library
                existing = self.global_library.get("interaction_definitions", {}).get(key, {})
                val = existing.get(data_key, "" if cell_type == "entry" else False)

                if cell_type == "bool":
                    var = tk.BooleanVar(value=bool(val))
                    widget = ttk.Checkbutton(frame, variable=var)
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

        # initial_biomass is a per-project FG override (not a library field).
        # Read from project FG entry first, then fall back to library default.
        if 'initial_biomass' in fg_entry and fg_entry.get('initial_biomass') is not None:
            init_b_val = fg_entry.get('initial_biomass')
        else:
            init_b_val = self.global_library.get("species_definitions", {}).get(fg_id, {}).get("initial_biomass", "")

        # Show the editor matching the FG category, hide the other.
        if category == "decision_makers":
            self.ndm_editor_frame.pack_forget()
            if not self.editor_frame.winfo_ismapped():
                self.editor_frame.pack(fill="x", padx=10, pady=5)
            self.editor_frame.configure(text=f"FG Editor ({self.fg_display(fg_id, include_sv=True)})")
            for key, var in self.prop_vars.items():
                if key == "initial_biomass":
                    var.set("" if init_b_val in (None, "") else str(init_b_val))
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
            self.ndm_editor_frame.configure(text=f"FG Editor ({self.fg_display(fg_id, include_sv=True)})")
            for key, var in self.ndm_prop_vars.items():
                if key == "initial_biomass":
                    var.set("" if init_b_val in (None, "") else str(init_b_val))
                    continue
                val = config.get(key, "")
                var.set(str(val))

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
        initial_biomass_val = None  # captured separately; stored per-project, not in library
        for key, var in prop_vars.items():
            val = var.get()
            if key == "initial_biomass":
                # Per-project FG override; do not write to global library.
                if val in (None, ""):
                    initial_biomass_val = None
                else:
                    try:
                        initial_biomass_val = float(val)
                    except ValueError:
                        initial_biomass_val = None
                continue
            if isinstance(var, tk.BooleanVar):
                config[key] = val
            else:
                try:
                    config[key] = float(val or 0)
                except ValueError:
                    config[key] = 0.0
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

        # Strip initial_biomass from library-bound config; it lives on the
        # project FG entry only.
        config.pop("initial_biomass", None)
        self.current_fg_configs[fg_id] = config

        # Persist initial_biomass on the project FG entry (per-project value).
        fg_entry = fgs[idx]
        if initial_biomass_val is None:
            fg_entry.pop("initial_biomass", None)
        else:
            fg_entry["initial_biomass"] = initial_biomass_val

        # Sync remaining fields with global library
        if "species_definitions" not in self.global_library:
            self.global_library["species_definitions"] = {}
        self.global_library["species_definitions"][fg_id] = config
        self.save_yaml(self.global_library, self.library_path)
        messagebox.showinfo("Success", f"Updated {fg_id}. Library updated; initial_biomass saved on project entry (remember to Save Project).")

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

        top = tk.Toplevel(self.root)
        top.title(f"Impact Table — {fg_name} × {impact_name}")
        top.transient(self.root)

        info = ttk.Label(
            top,
            text=("'Value' has an undefined range (physical impact value). "
                  "'Biomass factor' and 'Energy factor' must be in [0, 1]. "
                  "Linear interpolation between rows; outside the range, "
                  "the nearest endpoint value is used (no extrapolation)."),
            wraplength=480, justify="left",
        )
        info.pack(padx=10, pady=(10, 5), anchor="w")

        table_frame = ttk.Frame(top)
        table_frame.pack(padx=10, pady=5, fill="both", expand=True)

        headers = ("Value", "Biomass factor", "Energy factor")
        for j, h in enumerate(headers):
            ttk.Label(table_frame, text=h, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=j, padx=4, pady=2)

        row_vars = []  # list of (value_var, bf_var, ef_var)

        # Live validators: 'value' accepts any numeric (or partial) string; the
        # factor columns reject any intermediate string that cannot be the prefix
        # of a number in [0, 1] (so values outside the unit interval cannot even
        # be typed).
        def _validate_value(proposed):
            if proposed in ("", "-", "+", ".", "-.", "+."):
                return True
            try:
                float(proposed)
                return True
            except ValueError:
                return False

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
                lib_entry_initial = self.global_library.get("species_definitions", {}).get(fg_id, {}).get("initial_biomass")
                new_entry = {'group_id': fg_id}
                if lib_entry_initial is not None:
                    try:
                        new_entry['initial_biomass'] = float(lib_entry_initial)
                    except (TypeError, ValueError):
                        pass
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
            self.project_data.setdefault(category, []).append({'group_id': fg_id, 'initial_biomass': 0.0})
            self.current_fg_configs[fg_id] = {
                "display_name": fg_id,
                "is_decision_maker": is_dm,
                "growth_rate": 0.0,
                "max_energy_reserve": 0.0,
                "resting_metabolism": 0.0,
                "maintenance_level": 0.3,
                "movement_speed": 0.0,
                "movement_cost": 3.0,
                "feeding_cost": 3.0,
                "resting_cost": 1.0
            }
            # Add to global library immediately
            if "species_definitions" not in self.global_library:
                self.global_library["species_definitions"] = {}
            self.global_library["species_definitions"][fg_id] = self.current_fg_configs[fg_id]
            self.save_yaml(self.global_library, self.library_path)
            
            self.update_fg_list()
            self.refresh_matrix()

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

    def update_impact_list(self):
        def _impact_display(iv):
            impact_id = iv['impact_id']
            return self.global_library.get("impact_definitions", {}).get(impact_id, {}).get("display_name", impact_id)
        if self.project_data.get('impact_variables'):
            self.project_data['impact_variables'].sort(key=lambda iv: _impact_display(iv).lower())
        self.impact_listbox.delete(0, "end")
        for iv in self.project_data.get('impact_variables', []):
            self.impact_listbox.insert("end", _impact_display(iv))

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

    def new_project(self):
        self.project_data = {
            "project_metadata": {"name": "New Project"},
            "simulation_settings": {},
            "decision_makers": [],
            "non_decision_makers": [],
            "impact_variables": []
        }
        self.current_fg_configs = {}
        self.active_fg_category = None
        self.project_name_var.set("New Project")
        self.update_fg_list()
        self.update_impact_list()
        self.refresh_matrix()

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

    def save_project(self):
        if not self.project_path:
            self.project_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if self.project_path:
            self.project_data["project_metadata"]["name"] = self.project_name_var.get()
            self.save_yaml(self.project_data, self.project_path)
            self.add_to_recent(self.project_path)
            messagebox.showinfo("Success", f"Project saved to {self.project_path}")

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
    root = tk.Tk()
    app = FGConfigApp(root)
    root.mainloop()
