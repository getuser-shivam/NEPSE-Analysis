import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import time
import json
import os
import sys
from pathlib import Path
import queue
import webbrowser
from datetime import datetime

class AutoPromptGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto-Prompt Development System")
        self.root.geometry("1200x800")
        
        # Configuration
        self.config_file = "auto_prompt_config.json"
        self.execution_log = []
        self.is_running = False
        self.current_workflow_index = 0
        self.workflow_queue = queue.Queue()
        
        # Editors configuration
        self.editors = {
            "Windsurf": {
                "path": self.find_windsurf_executable(),
                "args": ["--workspace", "{workspace}"],
                "enabled": True
            },
            "VSCode": {
                "path": self.find_vscode_executable(),
                "args": ["{workspace}"],
                "enabled": False
            },
            "Cursor": {
                "path": self.find_cursor_executable(),
                "args": ["{workspace}"],
                "enabled": False
            }
        }
        
        # Auto-prompt workflows
        self.auto_workflows = [
            {"name": "review", "description": "Code review and bug checking", "enabled": True},
            {"name": "auto-prompt", "description": "Automatic enhancement prompting", "enabled": True},
            {"name": "auto-continuous", "description": "Continuous auto-enhancement", "enabled": True},
            {"name": "continuous", "description": "Never-stop enhancement", "enabled": False}
        ]
        
        self.setup_gui()
        self.load_config()
        self.discover_workflows()
        
    def find_windsurf_executable(self):
        """Find Windsurf executable"""
        if sys.platform == "win32":
            paths = [
                os.path.expanduser("~/AppData/Local/Programs/Windsurf/Windsurf.exe"),
                "C:\\Program Files\\Windsurf\\Windsurf.exe",
                "windsurf.exe"
            ]
        elif sys.platform == "darwin":
            paths = [
                "/Applications/Windsurf.app/Contents/MacOS/Windsurf",
                "windsurf"
            ]
        else:
            paths = ["windsurf"]
            
        for path in paths:
            if os.path.exists(path) or os.path.isabs(path):
                return path
        return "windsurf"  # fallback
    
    def find_vscode_executable(self):
        """Find VSCode executable"""
        if sys.platform == "win32":
            paths = [
                os.path.expanduser("~/AppData/Local/Programs/Microsoft VS Code/Code.exe"),
                "code.exe"
            ]
        elif sys.platform == "darwin":
            paths = [
                "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
                "code"
            ]
        else:
            paths = ["code"]
            
        for path in paths:
            if os.path.exists(path) or os.path.isabs(path):
                return path
        return "code"
    
    def find_cursor_executable(self):
        """Find Cursor executable"""
        if sys.platform == "win32":
            paths = [
                os.path.expanduser("~/AppData/Local/Programs/cursor/Cursor.exe"),
                "cursor.exe"
            ]
        elif sys.platform == "darwin":
            paths = [
                "/Applications/Cursor.app/Contents/MacOS/Cursor",
                "cursor"
            ]
        else:
            paths = ["cursor"]
            
        for path in paths:
            if os.path.exists(path) or os.path.isabs(path):
                return path
        return "cursor"
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 Auto-Prompt Development System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Editor selection
        ttk.Label(config_frame, text="Editor:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.editor_var = tk.StringVar()
        for i, (editor_name, editor_config) in enumerate(self.editors.items()):
            rb = ttk.Radiobutton(config_frame, text=editor_name, variable=self.editor_var, 
                                value=editor_name, command=self.on_editor_change)
            rb.grid(row=i+1, column=0, sticky=tk.W, padx=(20, 5))
            
            # Status indicator
            status = "✅" if os.path.exists(editor_config["path"]) else "❌"
            ttk.Label(config_frame, text=status).grid(row=i+1, column=1, sticky=tk.W)
        
        # Workspace selection
        ttk.Label(config_frame, text="Workspace:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(20, 5))
        
        workspace_frame = ttk.Frame(config_frame)
        workspace_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.workspace_var = tk.StringVar(value=os.getcwd())
        self.workspace_entry = ttk.Entry(workspace_frame, textvariable=self.workspace_var, width=30)
        self.workspace_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(workspace_frame, text="Browse", command=self.browse_workspace).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Auto-prompt workflows
        ttk.Label(config_frame, text="Auto-Prompt Workflows:", font=("Arial", 10, "bold")).grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        
        self.workflow_vars = {}
        for i, workflow in enumerate(self.auto_workflows):
            var = tk.BooleanVar(value=workflow["enabled"])
            self.workflow_vars[workflow["name"]] = var
            
            cb = ttk.Checkbutton(config_frame, text=f"{workflow['name']} - {workflow['description']}", 
                                variable=var)
            cb.grid(row=8+i, column=0, columnspan=2, sticky=tk.W, padx=(20, 5))
        
        # Execution settings
        ttk.Label(config_frame, text="Execution Settings:", font=("Arial", 10, "bold")).grid(row=12, column=0, sticky=tk.W, pady=(20, 5))
        
        # Loop mode
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Loop Continuously", variable=self.loop_var).grid(row=13, column=0, columnspan=2, sticky=tk.W, padx=(20, 5))
        
        # Delay between workflows
        ttk.Label(config_frame, text="Delay (seconds):").grid(row=14, column=0, sticky=tk.W, pady=(10, 5))
        self.delay_var = tk.IntVar(value=30)
        ttk.Spinbox(config_frame, from_=5, to=300, textvariable=self.delay_var, width=10).grid(row=14, column=1, sticky=tk.W)
        
        # Middle panel - Control
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, pady=(0, 20))
        
        # Control buttons
        self.start_button = ttk.Button(control_frame, text="🚀 Start Auto-Prompt", 
                                      command=self.start_auto_prompt, style="Accent.TButton")
        self.start_button.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop", 
                                     command=self.stop_auto_prompt, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="🔄 Restart", command=self.restart_auto_prompt).grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # Quick actions
        ttk.Label(control_frame, text="Quick Actions:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        
        ttk.Button(control_frame, text="📝 Open Editor", command=self.open_editor).grid(row=6, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="🔍 Discover Workflows", command=self.discover_workflows).grid(row=7, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="💾 Save Config", command=self.save_config).grid(row=8, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="📊 Show Stats", command=self.show_stats).grid(row=9, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Progress
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=10, column=0, sticky=(tk.W, tk.E), pady=20)
        
        ttk.Label(control_frame, text="Progress:", font=("Arial", 10, "bold")).grid(row=11, column=0, sticky=tk.W, pady=(10, 5))
        
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(row=12, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(control_frame, mode='determinate')
        self.progress_bar.grid(row=13, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Current workflow
        ttk.Label(control_frame, text="Current:", font=("Arial", 10, "bold")).grid(row=14, column=0, sticky=tk.W, pady=(10, 5))
        self.current_workflow_var = tk.StringVar(value="None")
        ttk.Label(control_frame, textvariable=self.current_workflow_var).grid(row=15, column=0, sticky=tk.W)
        
        # Bottom panel - Output
        output_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.output_text.tag_configure("info", foreground="blue")
        self.output_text.tag_configure("success", foreground="green")
        self.output_text.tag_configure("error", foreground="red")
        self.output_text.tag_configure("warning", foreground="orange")
        
    def browse_workspace(self):
        """Browse for workspace directory"""
        directory = filedialog.askdirectory(initialdir=self.workspace_var.get())
        if directory:
            self.workspace_var.set(directory)
    
    def on_editor_change(self):
        """Handle editor selection change"""
        editor_name = self.editor_var.get()
        if editor_name in self.editors:
            editor_config = self.editors[editor_name]
            self.log(f"Selected editor: {editor_name}", "info")
            if not os.path.exists(editor_config["path"]):
                self.log(f"Warning: {editor_name} executable not found at {editor_config['path']}", "warning")
    
    def discover_workflows(self):
        """Discover available workflows"""
        workflow_dir = Path(".windsurf/workflows")
        if not workflow_dir.exists():
            self.log("No .windsurf/workflows directory found", "warning")
            return
        
        workflows = []
        for file in workflow_dir.glob("*.md"):
            workflows.append(file.stem)
        
        self.log(f"Discovered {len(workflows)} workflows: {', '.join(workflows)}", "info")
        
        # Update auto-workflows list
        for workflow in workflows:
            if workflow not in [w["name"] for w in self.auto_workflows]:
                self.auto_workflows.append({
                    "name": workflow,
                    "description": f"Workflow: {workflow}",
                    "enabled": False
                })
                var = tk.BooleanVar(value=False)
                self.workflow_vars[workflow] = var
    
    def start_auto_prompt(self):
        """Start the auto-prompt sequence"""
        if self.is_running:
            return
        
        # Get selected workflows
        selected_workflows = []
        for workflow in self.auto_workflows:
            if self.workflow_vars.get(workflow["name"], tk.BooleanVar()).get():
                selected_workflows.append(workflow["name"])
        
        if not selected_workflows:
            messagebox.showwarning("No Workflows", "Please select at least one workflow to run.")
            return
        
        # Check editor
        editor_name = self.editor_var.get()
        if not editor_name:
            messagebox.showwarning("No Editor", "Please select an editor.")
            return
        
        # Start the sequence
        self.is_running = True
        self.selected_workflows = selected_workflows
        self.current_workflow_index = 0
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running")
        
        # Open editor
        self.open_editor()
        
        # Start execution thread
        threading.Thread(target=self.execute_workflow_sequence, daemon=True).start()
        
        self.log("🚀 Auto-prompt sequence started!", "success")
    
    def stop_auto_prompt(self):
        """Stop the auto-prompt sequence"""
        self.is_running = False
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
        self.log("⏹️ Auto-prompt sequence stopped", "warning")
    
    def restart_auto_prompt(self):
        """Restart the auto-prompt sequence"""
        self.stop_auto_prompt()
        time.sleep(1)
        self.start_auto_prompt()
    
    def open_editor(self):
        """Open the selected editor"""
        editor_name = self.editor_var.get()
        if not editor_name or editor_name not in self.editors:
            return
        
        editor_config = self.editors[editor_name]
        workspace = self.workspace_var.get()
        
        try:
            # Prepare arguments
            args = [arg.format(workspace=workspace) for arg in editor_config["args"]]
            cmd = [editor_config["path"]] + args
            
            self.log(f"Opening {editor_name} with workspace: {workspace}", "info")
            
            # Start editor
            subprocess.Popen(cmd, shell=True)
            
            # Give it time to open
            time.sleep(3)
            
        except Exception as e:
            self.log(f"Failed to open {editor_name}: {e}", "error")
    
    def execute_workflow_sequence(self):
        """Execute the workflow sequence"""
        while self.is_running:
            try:
                # Get current workflow
                if self.current_workflow_index >= len(self.selected_workflows):
                    if self.loop_var.get():
                        self.current_workflow_index = 0
                        self.log("🔄 Looping back to first workflow", "info")
                    else:
                        self.log("✅ All workflows completed!", "success")
                        self.root.after(0, self.stop_auto_prompt)
                        break
                
                workflow_name = self.selected_workflows[self.current_workflow_index]
                
                # Update progress
                total = len(self.selected_workflows)
                current = self.current_workflow_index + 1
                progress = (current / total) * 100
                
                self.root.after(0, lambda: self.update_progress(current, total, workflow_name, progress))
                
                # Execute workflow
                self.log(f"🔄 Executing workflow: {workflow_name}", "info")
                success = self.execute_workflow(workflow_name)
                
                if success:
                    self.log(f"✅ Workflow {workflow_name} completed successfully", "success")
                else:
                    self.log(f"❌ Workflow {workflow_name} failed", "error")
                
                # Move to next workflow
                self.current_workflow_index += 1
                
                # Wait before next workflow
                if self.is_running and self.current_workflow_index < len(self.selected_workflows):
                    delay = self.delay_var.get()
                    self.log(f"⏳ Waiting {delay} seconds before next workflow...", "info")
                    time.sleep(delay)
                
            except Exception as e:
                self.log(f"Error in workflow sequence: {e}", "error")
                break
    
    def execute_workflow(self, workflow_name):
        """Execute a single workflow"""
        try:
            # Use windsurf_trigger.py to execute workflow
            cmd = [
                sys.executable, "windsurf_trigger.py",
                "--workflow", workflow_name,
                "--workspace", self.workspace_var.get()
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Log output
            if result.stdout:
                self.root.after(0, lambda: self.log_output(result.stdout, "info"))
            
            if result.stderr:
                self.root.after(0, lambda: self.log_output(result.stderr, "error"))
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            self.log(f"Workflow {workflow_name} timed out", "error")
            return False
        except Exception as e:
            self.log(f"Error executing workflow {workflow_name}: {e}", "error")
            return False
    
    def update_progress(self, current, total, workflow_name, progress):
        """Update progress indicators"""
        self.progress_var.set(f"{current}/{total}")
        self.progress_bar['value'] = progress
        self.current_workflow_var.set(workflow_name)
    
    def log(self, message, tag="info"):
        """Log a message to the output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.output_text.see(tk.END)
        self.output_text.update()
    
    def log_output(self, output, tag="info"):
        """Log workflow output"""
        for line in output.strip().split('\n'):
            if line.strip():
                self.log(f"  {line}", tag)
    
    def save_config(self):
        """Save configuration to file"""
        config = {
            "editor": self.editor_var.get(),
            "workspace": self.workspace_var.get(),
            "workflows": {name: var.get() for name, var in self.workflow_vars.items()},
            "loop": self.loop_var.get(),
            "delay": self.delay_var.get()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.log("✅ Configuration saved", "success")
        except Exception as e:
            self.log(f"Failed to save config: {e}", "error")
    
    def load_config(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if "editor" in config:
                self.editor_var.set(config["editor"])
            
            if "workspace" in config:
                self.workspace_var.set(config["workspace"])
            
            if "workflows" in config:
                for name, enabled in config["workflows"].items():
                    if name in self.workflow_vars:
                        self.workflow_vars[name].set(enabled)
            
            if "loop" in config:
                self.loop_var.set(config["loop"])
            
            if "delay" in config:
                self.delay_var.set(config["delay"])
            
            self.log("✅ Configuration loaded", "success")
            
        except Exception as e:
            self.log(f"Failed to load config: {e}", "error")
    
    def show_stats(self):
        """Show execution statistics"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Execution Statistics")
        stats_window.geometry("400x300")
        
        # Create stats display
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Calculate stats
        total_executions = len(self.execution_log)
        successful = sum(1 for log in self.execution_log if log.get("success", False))
        failed = total_executions - successful
        
        stats_text.insert(tk.END, "📊 Execution Statistics\n")
        stats_text.insert(tk.END, "=" * 30 + "\n\n")
        stats_text.insert(tk.END, f"Total Executions: {total_executions}\n")
        stats_text.insert(tk.END, f"Successful: {successful}\n")
        stats_text.insert(tk.END, f"Failed: {failed}\n")
        stats_text.insert(tk.END, f"Success Rate: {(successful/total_executions*100):.1f}%\n" if total_executions > 0 else "Success Rate: N/A\n")
        
        if self.execution_log:
            stats_text.insert(tk.END, "\nRecent Executions:\n")
            stats_text.insert(tk.END, "-" * 30 + "\n")
            for log in self.execution_log[-10:]:
                timestamp = log.get("timestamp", "Unknown")
                workflow = log.get("workflow", "Unknown")
                status = "✅" if log.get("success", False) else "❌"
                stats_text.insert(tk.END, f"{timestamp} {status} {workflow}\n")

def main():
    root = tk.Tk()
    app = AutoPromptGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
