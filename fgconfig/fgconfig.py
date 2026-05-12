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
        self.project_data = {"project_metadata": {"name": "New Project"}, "simulation_settings": {}, "functional_groups": [], "impact_variables": []}
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
        """Handle mouse wheel and trackpad scroll events."""
        # Determine scroll direction
        if event.num == 4: # Linux scroll up
            delta = -1
        elif event.num == 5: # Linux scroll down
            delta = 1
        elif event.delta: # Windows/macOS
            delta = int(-1 * (event.delta / 120))
        else:
            return

        # Apply scroll to matrix canvas if it exists
        if hasattr(self, 'matrix_canvas') and self.matrix_canvas.winfo_exists():
            # Check if the matrix tab is visible or if the event happened over the canvas
            if self.notebook.index(self.notebook.select()) == 1:
                self.matrix_canvas.yview_scroll(delta, "units")

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

        # Tab 2: Interaction Matrix
        self.matrix_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_tab, text="Interaction Matrix")
        self.setup_matrix_tab()

    def setup_project_tab(self):
        # Project Info
        info_frame = ttk.LabelFrame(self.project_tab, text="Project Info")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(info_frame, text="Project Name:").grid(row=0, column=0, sticky="w", padx=5)
        self.project_name_var = tk.StringVar(value="New Project")
        ttk.Entry(info_frame, textvariable=self.project_name_var).grid(row=0, column=1, sticky="ew", padx=5)

        # FG List in Project
        fg_frame = ttk.LabelFrame(self.project_tab, text="Functional Groups in Project")
        fg_frame.pack(expand=True, fill="both", padx=10, pady=5)

        self.fg_listbox = tk.Listbox(fg_frame)
        self.fg_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        self.fg_listbox.bind("<<ListboxSelect>>", self.on_fg_select)

        btn_frame = ttk.Frame(fg_frame)
        btn_frame.pack(side="right", fill="y", padx=5, pady=5)

        ttk.Button(btn_frame, text="Add from Library", command=self.add_from_library).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add New FG", command=self.add_new_fg).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Remove FG", command=self.remove_fg).pack(fill="x", pady=2)

        # Impact Variables list in project
        impact_frame = ttk.LabelFrame(self.project_tab, text="Impact Variables in Project")
        impact_frame.pack(expand=True, fill="both", padx=10, pady=5)

        self.impact_listbox = tk.Listbox(impact_frame)
        self.impact_listbox.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        impact_btn_frame = ttk.Frame(impact_frame)
        impact_btn_frame.pack(side="right", fill="y", padx=5, pady=5)

        ttk.Button(impact_btn_frame, text="Add from Library", command=self.add_impact_from_library).pack(fill="x", pady=2)
        ttk.Button(impact_btn_frame, text="Remove Impact", command=self.remove_impact).pack(fill="x", pady=2)

        # FG Editor
        self.editor_frame = ttk.LabelFrame(self.project_tab, text="FG Editor")
        self.editor_frame.pack(fill="x", padx=10, pady=5)
        
        self.prop_vars = {}
        main_props = [
            ("Is Decision Maker", "is_decision_maker", "check"),
            ("Growth Rate (GR_X)", "growth_rate", "entry"),
            ("Max Energy Reserve (ME_X MJ/ton)", "max_energy_reserve", "entry"),
            ("Energy Content (MJ/ton)", "energy_content", "entry"),
            ("Resting Metabolism (MJ/ton)", "resting_metabolism", "entry"),
            ("Movement Speed (cells/tick)", "movement_speed", "entry")
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

    def setup_matrix_tab(self):
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
        self.matrix_canvas.configure(yscrollcommand=self.matrix_scrollbar.set)

        self.matrix_canvas.pack(side="left", expand=True, fill="both")
        self.matrix_scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel scrolling to the canvas and all its potential children
        self.matrix_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.matrix_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.matrix_canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        self.refresh_matrix()

    def refresh_matrix(self):
        for widget in self.matrix_container.winfo_children():
            widget.destroy()

        fgs = [fg['group_id'] for fg in self.project_data['functional_groups']]
        if not fgs:
            ttk.Label(self.matrix_container, text="Add FGs to the project to see interaction matrices.").pack(padx=10, pady=10)
            return

        self.matrix_entries = {}
        self.matrix_widgets = {}
        
        # Matrix 1: Predation (boolean: row eats column; cannibalism allowed on diagonal)
        self.create_matrix_section("Predation (row eats column)", "preys_on", fgs, fgs, cell_type="bool")

        # Matrix 2: Max Intake
        self.create_matrix_section("Max Intake Rate (I_XY) [ton prey / ton consumer]", "max_intake_rate", fgs, fgs)
        
        # Matrix 3: Impact Sensitivity (0..1 ratio with slider)
        impacts = [iv['impact_id'] for iv in self.project_data.get('impact_variables', [])]
        if impacts:
            self.create_matrix_section("Impact Sensitivity [0..1 = 0%..100%]", "impact_sensitivity", fgs, impacts, cell_type="ratio")

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

        ttk.Button(self.matrix_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)

    def create_matrix_section(self, title, data_key, row_ids, col_ids, cell_type="entry"):
        frame = ttk.LabelFrame(self.matrix_container, text=title)
        frame.pack(fill="x", padx=10, pady=10)

        # Headers
        ttk.Label(frame, text="Group \\ Var").grid(row=0, column=0, padx=5, pady=5)
        for j, col_id in enumerate(col_ids):
            # Use display_name from impact_definitions if available
            if data_key == "impact_sensitivity":
                label_text = self.global_library.get("impact_definitions", {}).get(col_id, {}).get("display_name", col_id)
            else:
                label_text = self.fg_display(col_id)
            ttk.Label(frame, text=label_text).grid(row=0, column=j+1, padx=5, pady=5)

        for i, row_id in enumerate(row_ids):
            row_label = self.fg_display(row_id)
            ttk.Label(frame, text=row_label).grid(row=i+1, column=0, padx=5, pady=5)
            for j, col_id in enumerate(col_ids):
                # Unique key for storage
                if data_key == "impact_sensitivity":
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
                elif cell_type == "ratio":
                    # Clamp initial value to [0, 1]
                    try:
                        init_val = float(val) if val not in ("", None, False) else 0.0
                    except (TypeError, ValueError):
                        init_val = 0.0
                    init_val = max(0.0, min(1.0, init_val))
                    var = tk.StringVar(value=f"{init_val:.2f}")
                    cell = ttk.Frame(frame)
                    cell.grid(row=i+1, column=j+1, padx=2, pady=2)
                    entry = ttk.Entry(cell, textvariable=var, width=5)
                    entry.pack(side="left")
                    # Use DoubleVar for slider to avoid feedback loops
                    slider_var = tk.DoubleVar(value=init_val)
                    slider = ttk.Scale(cell, from_=0.0, to=1.0, orient="horizontal",
                                       variable=slider_var, length=80)
                    slider.pack(side="left", padx=(2, 0))
                    # Sync slider -> entry
                    def _on_slider(*_a, sv=slider_var, tv=var):
                        tv.set(f"{sv.get():.2f}")
                    slider_var.trace_add("write", _on_slider)
                    # Sync entry -> slider (clamp)
                    def _on_entry(*_a, sv=slider_var, tv=var):
                        try:
                            v = float(tv.get())
                        except (TypeError, ValueError):
                            return
                        v = max(0.0, min(1.0, v))
                        if abs(sv.get() - v) > 1e-9:
                            sv.set(v)
                    var.trace_add("write", _on_entry)
                    widget = entry
                else:
                    var = tk.StringVar(value=str(val))
                    widget = ttk.Entry(frame, textvariable=var, width=10)
                    widget.grid(row=i+1, column=j+1, padx=2, pady=2)
                self.matrix_entries[key][data_key] = var
                if not hasattr(self, "matrix_widgets"):
                    self.matrix_widgets = {}
                if key not in self.matrix_widgets:
                    self.matrix_widgets[key] = {}
                self.matrix_widgets[key][data_key] = widget

    def on_fg_select(self, event):
        selection = self.fg_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        fgs = self.project_data['functional_groups']
        if idx >= len(fgs):
            return
        fg_id = fgs[idx]['group_id']
        config = self.current_fg_configs.get(fg_id, {})
        # Update editor title to include the FG name (English/Swedish)
        self.editor_frame.configure(text=f"FG Editor ({self.fg_display(fg_id, include_sv=True)})")
        
        for key, var in self.prop_vars.items():
            val = config.get(key, "")
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set(str(val))

    def apply_fg_changes(self):
        selection = self.fg_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a Functional Group from the list.")
            return
        idx = selection[0]
        fgs = self.project_data['functional_groups']
        if idx >= len(fgs):
            return
        fg_id = fgs[idx]['group_id']
        # Preserve existing display_name from current config / library
        existing = self.current_fg_configs.get(fg_id, {})
        display_name = existing.get("display_name") if isinstance(existing, dict) else None
        if not display_name:
            display_name = self.global_library.get("species_definitions", {}).get(fg_id, {}).get("display_name", fg_id)
        config = {"display_name": display_name}
        
        for key, var in self.prop_vars.items():
            val = var.get()
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
                    
        self.current_fg_configs[fg_id] = config
        
        # Sync with global library
        if "species_definitions" not in self.global_library:
            self.global_library["species_definitions"] = {}
        self.global_library["species_definitions"][fg_id] = config
        self.save_yaml(self.global_library, self.library_path)
        messagebox.showinfo("Success", f"Updated {fg_id} and saved to library.")

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
                            if data_key == "impact_sensitivity":
                                val = max(0.0, min(1.0, val))
                            self.global_library["interaction_definitions"][key][data_key] = val
                        except ValueError:
                            pass # skip invalid
        
        self.save_yaml(self.global_library, self.library_path)
        messagebox.showinfo("Success", "Updated interactions and saved to library.")

    def add_from_library(self):
        lib_fgs = list(self.global_library.get("species_definitions", {}).keys())
        if not lib_fgs:
            messagebox.showinfo("Library Empty", "Global library is empty.")
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
                if not any(fg['group_id'] == fg_id for fg in self.project_data['functional_groups']):
                    self.project_data['functional_groups'].append({'group_id': fg_id})
                    self.current_fg_configs[fg_id] = self.global_library["species_definitions"][fg_id]
                    added += 1
            self.update_fg_list()
            self.refresh_matrix()
            top.destroy()

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(side="left")
        ttk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side="right")
        ttk.Button(btn_frame, text="Add", command=do_add).pack(side="right", padx=5)
        lb.bind("<Double-Button-1>", lambda e: do_add())

    def add_new_fg(self):
        import tkinter.simpledialog as sd
        fg_id = sd.askstring("New FG", "Enter ID for new Functional Group (English):")
        if fg_id:
            if any(fg['group_id'] == fg_id for fg in self.project_data['functional_groups']):
                messagebox.showerror("Error", "FG ID already exists in project.")
                return
            
            self.project_data['functional_groups'].append({'group_id': fg_id})
            self.current_fg_configs[fg_id] = {
                "display_name": fg_id,
                "is_decision_maker": False,
                "growth_rate": 0.0,
                "max_energy_reserve": 0.0,
                "resting_metabolism": 0.0,
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

    def remove_fg(self):
        selection = self.fg_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        fgs = self.project_data['functional_groups']
        if idx >= len(fgs):
            return
        fg_id = fgs[idx]['group_id']
        self.project_data['functional_groups'] = [fg for fg in fgs if fg['group_id'] != fg_id]
        if fg_id in self.current_fg_configs:
            del self.current_fg_configs[fg_id]
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
        self.fg_listbox.delete(0, "end")
        for fg in self.project_data['functional_groups']:
            self.fg_listbox.insert("end", self.fg_display(fg['group_id']))

    def update_impact_list(self):
        self.impact_listbox.delete(0, "end")
        for iv in self.project_data.get('impact_variables', []):
            impact_id = iv['impact_id']
            display = self.global_library.get("impact_definitions", {}).get(impact_id, {}).get("display_name", impact_id)
            self.impact_listbox.insert("end", display)

    def add_impact_from_library(self):
        lib_impacts = list(self.global_library.get("impact_definitions", {}).keys())
        if not lib_impacts:
            messagebox.showinfo("Library Empty", "No impact variables in global library.")
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
        self.project_data = {"project_metadata": {"name": "New Project"}, "simulation_settings": {}, "functional_groups": [], "impact_variables": []}
        self.current_fg_configs = {}
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
            # Load configs for active FGs
            self.current_fg_configs = {}
            for fg in self.project_data.get('functional_groups', []):
                gid = fg['group_id']
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
