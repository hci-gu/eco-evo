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
        self.project_path = None
        
        self.global_library = self.load_yaml(self.library_path) or {"species_definitions": {}, "interaction_definitions": {}}
        self.project_data = {"project_metadata": {"name": "New Project"}, "simulation_settings": {}, "functional_groups": []}
        self.current_fg_configs = {} # local project configs for active FGs

        self.setup_ui()

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
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

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

        # FG Editor
        self.editor_frame = ttk.LabelFrame(self.project_tab, text="FG Editor")
        self.editor_frame.pack(fill="x", padx=10, pady=5)
        
        self.prop_vars = {}
        props = [
            ("Display Name", "display_name", "entry"),
            ("Is Decision Maker", "is_decision_maker", "check"),
            ("Growth Rate (GR_X)", "growth_rate", "entry"),
            ("Max Energy Reserve (ME_X MJ/ton)", "max_energy_reserve", "entry"),
            ("Resting Metabolism (MJ/ton)", "resting_metabolism", "entry"),
            ("Movement Speed (cells/tick)", "movement_speed", "entry"),
            ("Movement Cost (MJ/ton)", "movement_cost", "entry"),
            ("Feeding Cost (MJ/ton)", "feeding_cost", "entry")
        ]
        
        for i, (label, key, type) in enumerate(props):
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

        ttk.Button(self.editor_frame, text="Apply Changes", command=self.apply_fg_changes).grid(row=len(props), column=0, columnspan=2, pady=5)

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
        self.refresh_matrix()

    def refresh_matrix(self):
        for widget in self.matrix_container.winfo_children():
            widget.destroy()

        fgs = [fg['group_id'] for fg in self.project_data['functional_groups']]
        if not fgs:
            ttk.Label(self.matrix_container, text="Add FGs to the project to see interaction matrices.").pack(padx=10, pady=10)
            return

        self.matrix_entries = {}
        
        # Matrix 1: Max Intake
        self.create_matrix_section("Max Intake Rate (I_XY) [ton prey / ton consumer]", "max_intake_rate", fgs, fgs)
        
        # Matrix 2: Energy Gain
        self.create_matrix_section("Energy Gain [MJ / ton prey]", "energy_gain", fgs, fgs)
        
        # Matrix 3: Impact Sensitivity
        impacts = ["Bottom Trawling", "Pelagic Trawling", "Hunting", "Logging", "Chemicals", "Windfarm Noise", "Ship Traffic", "Turbidity"]
        self.create_matrix_section("Impact Sensitivity [MJ loss / unit impact]", "impact_sensitivity", fgs, impacts)

        ttk.Button(self.matrix_container, text="Apply All Matrix Changes", command=self.apply_matrix_changes).pack(pady=10)

    def create_matrix_section(self, title, data_key, row_ids, col_ids):
        frame = ttk.LabelFrame(self.matrix_container, text=title)
        frame.pack(fill="x", padx=10, pady=10)

        # Headers
        ttk.Label(frame, text="Group \ Var").grid(row=0, column=0, padx=5, pady=5)
        for j, col_id in enumerate(col_ids):
            ttk.Label(frame, text=col_id).grid(row=0, column=j+1, padx=5, pady=5)

        for i, row_id in enumerate(row_ids):
            ttk.Label(frame, text=row_id).grid(row=i+1, column=0, padx=5, pady=5)
            for j, col_id in enumerate(col_ids):
                # Unique key for storage
                if data_key == "impact_sensitivity":
                    key = f"{row_id}_impacted_by_{col_id.lower().replace(' ', '_')}"
                else:
                    key = f"{row_id}_preys_on_{col_id}"
                
                if key not in self.matrix_entries:
                    self.matrix_entries[key] = {}

                # Look for existing value in library
                val = ""
                if key in self.global_library.get("interaction_definitions", {}):
                    val = self.global_library["interaction_definitions"][key].get(data_key, "")
                
                var = tk.StringVar(value=str(val))
                ent = ttk.Entry(frame, textvariable=var, width=10)
                ent.grid(row=i+1, column=j+1, padx=2, pady=2)
                self.matrix_entries[key][data_key] = var

    def on_fg_select(self, event):
        selection = self.fg_listbox.curselection()
        if not selection:
            return
        fg_id = self.fg_listbox.get(selection[0])
        config = self.current_fg_configs.get(fg_id, {})
        
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
        
        fg_id = self.fg_listbox.get(selection[0])
        config = {"display_name": self.prop_vars["display_name"].get()}
        
        for key, var in self.prop_vars.items():
            if key == "display_name": continue
            val = var.get()
            if isinstance(var, tk.BooleanVar):
                config[key] = val
            else:
                try:
                    config[key] = float(val or 0)
                except ValueError:
                    config[key] = 0.0
                    
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
                val_str = var.get()
                if val_str:
                    try:
                        val = float(val_str)
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
        lb = tk.Listbox(top, selectmode="multiple")
        lb.pack(padx=10, pady=10)
        for item in lib_fgs:
            lb.insert("end", item)
        
        def do_add():
            for i in lb.curselection():
                fg_id = lb.get(i)
                if not any(fg['group_id'] == fg_id for fg in self.project_data['functional_groups']):
                    self.project_data['functional_groups'].append({'group_id': fg_id})
                    self.current_fg_configs[fg_id] = self.global_library["species_definitions"][fg_id]
            self.update_fg_list()
            self.refresh_matrix()
            top.destroy()

        ttk.Button(top, text="Add", command=do_add).pack(pady=5)

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
                "movement_cost": 0.0,
                "feeding_cost": 0.0
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
        fg_id = self.fg_listbox.get(selection[0])
        self.project_data['functional_groups'] = [fg for fg in self.project_data['functional_groups'] if fg['group_id'] != fg_id]
        if fg_id in self.current_fg_configs:
            del self.current_fg_configs[fg_id]
        self.update_fg_list()
        self.refresh_matrix()

    def update_fg_list(self):
        self.fg_listbox.delete(0, "end")
        for fg in self.project_data['functional_groups']:
            self.fg_listbox.insert("end", fg['group_id'])

    def new_project(self):
        self.project_data = {"project_metadata": {"name": "New Project"}, "simulation_settings": {}, "functional_groups": []}
        self.current_fg_configs = {}
        self.project_name_var.set("New Project")
        self.update_fg_list()
        self.refresh_matrix()

    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if path:
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
                self.update_fg_list()
                self.refresh_matrix()

    def save_project(self):
        if not self.project_path:
            self.project_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if self.project_path:
            self.project_data["project_metadata"]["name"] = self.project_name_var.get()
            self.save_yaml(self.project_data, self.project_path)
            messagebox.showinfo("Success", f"Project saved to {self.project_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FGConfigApp(root)
    root.mainloop()
