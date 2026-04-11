#!/usr/bin/env python3
"""
Windsurf Workflow Trigger Panel - Integrated GUI
A GUI panel that integrates with Windsurf editor for workflow triggering
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import os
import sys
import json
import logging
from datetime import datetime
import time
import platform
from pathlib import Path


class WindsurfWorkflowPanel:
    """Integrated workflow trigger panel for Windsurf editor"""

    def __init__(self, root=None):
        # Create root window if not provided
        if root is None:
            self.root = tk.Tk()
            self.root.title("Windsurf Workflow Trigger Panel")
            self.root.geometry("900x700")
            self.root.resizable(True, True)
        else:
            self.root = root

        # Initialize logging
        self.setup_logging()

        # Initialize variables
        self.workflow_history = []
        self.active_processes = {}
        self.current_workspace = self._get_current_workspace()
        self.windsurf_trigger = self._create_windsurf_trigger()

        # Create the main interface
        self.create_main_interface()

        # Load workflow history
        self.load_history()

        # Auto-refresh workflows
        self.refresh_workflows()

        self.logger.info("Windsurf Workflow Panel initialized")

        # Start the main loop if we created our own root
        if root is None:
            self.root.mainloop()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('windsurf_workflow_panel.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('WindsurfWorkflowPanel')

    def _get_current_workspace(self):
        """Get the current workspace path"""
        try:
            # Try to get from environment or current directory
            workspace = os.environ.get('WINDSURF_WORKSPACE') or os.getcwd()

            # Look for workspace indicators
            workspace_indicators = ['.windsurf', '.git', 'main.py', 'requirements.txt', 'package.json']

            current_path = Path(workspace)
            for path in [current_path] + list(current_path.parents):
                if any((path / indicator).exists() for indicator in workspace_indicators):
                    return str(path)

            return workspace

        except Exception as e:
            self.logger.error(f"Failed to detect workspace: {e}")
            return os.getcwd()

    def _create_windsurf_trigger(self):
        """Create and return a WindsurfTrigger instance"""
        try:
            # Import the WindsurfTrigger class
            import windsurf_trigger
            return windsurf_trigger.WindsurfTrigger(workspace_path=self.current_workspace)
        except ImportError:
            self.logger.warning("windsurf_trigger module not found, some features disabled")
            return None

    def create_main_interface(self):
        """Create the main GUI interface"""
        # Main container
        main_container = ttk.Frame(self.root, padding="15")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        header_frame.columnconfigure(1, weight=1)

        ttk.Label(header_frame, text="🎯", font=('Arial', 20)).grid(row=0, column=0, padx=(0, 10))
        ttk.Label(header_frame, text="Windsurf Workflow Trigger Panel",
                 font=('Arial', 16, 'bold')).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(header_frame, text=f"Workspace: {self.current_workspace}",
                 font=('Arial', 9)).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))

        # Main content area with paned window
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Workflow controls
        left_panel = ttk.Frame(paned, padding="10")
        paned.add(left_panel, weight=1)

        # Right panel - Output and history
        right_panel = ttk.Frame(paned, padding="10")
        paned.add(right_panel, weight=2)

        # Create left panel content
        self.create_workflow_controls(left_panel)

        # Create right panel content
        self.create_output_panel(right_panel)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def create_workflow_controls(self, parent):
        """Create the workflow control panel"""
        # Workflow discovery section
        discovery_frame = ttk.LabelFrame(parent, text="🔍 Workflow Discovery", padding="10")
        discovery_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        discovery_frame.columnconfigure(0, weight=1)

        ttk.Button(discovery_frame, text="🔄 Refresh Workflows",
                  command=self.refresh_workflows).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Workspace selector
        workspace_frame = ttk.Frame(discovery_frame)
        workspace_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        workspace_frame.columnconfigure(1, weight=1)

        ttk.Label(workspace_frame, text="Workspace:").grid(row=0, column=0, padx=(0, 5))
        self.workspace_var = tk.StringVar(value=self.current_workspace)
        workspace_entry = ttk.Entry(workspace_frame, textvariable=self.workspace_var)
        workspace_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(workspace_frame, text="📁 Browse",
                  command=self.browse_workspace).grid(row=0, column=2)

        # Workflow list section
        list_frame = ttk.LabelFrame(parent, text="📋 Available Workflows", padding="10")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Workflow listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.workflow_listbox = tk.Listbox(
            listbox_frame,
            height=12,
            selectmode=tk.MULTIPLE,
            font=('Courier', 9),
            yscrollcommand=scrollbar.set
        )
        self.workflow_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.workflow_listbox.yview)

        # Bind events
        self.workflow_listbox.bind('<<ListboxSelect>>', self.on_workflow_select)
        self.workflow_listbox.bind('<Double-1>', lambda e: self.trigger_selected_workflows())

        # Workflow description
        desc_frame = ttk.Frame(list_frame)
        desc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(desc_frame, text="Selected:").grid(row=0, column=0, sticky=tk.W)
        self.selected_count_var = tk.StringVar(value="0 workflows")
        ttk.Label(desc_frame, textvariable=self.selected_count_var,
                 font=('Arial', 9, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Trigger controls section
        trigger_frame = ttk.LabelFrame(parent, text="🚀 Workflow Execution", padding="10")
        trigger_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        trigger_frame.columnconfigure(0, weight=1)

        # Execution mode
        mode_frame = ttk.Frame(trigger_frame)
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.execution_mode = tk.StringVar(value="async")
        ttk.Radiobutton(mode_frame, text="⚡ Async (Fire & Forget)",
                       variable=self.execution_mode, value="async").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(mode_frame, text="⏳ Sync (Wait for Completion)",
                       variable=self.execution_mode, value="sync").grid(row=1, column=0, sticky=tk.W)

        # Timeout setting
        timeout_frame = ttk.Frame(trigger_frame)
        timeout_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(timeout_frame, text="Timeout (seconds):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.timeout_var = tk.IntVar(value=300)
        ttk.Spinbox(timeout_frame, from_=30, to=3600, textvariable=self.timeout_var,
                   width=8).grid(row=0, column=1, sticky=tk.W)

        # Trigger buttons
        button_frame = ttk.Frame(trigger_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="🎯 Trigger Selected",
                  command=self.trigger_selected_workflows).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="🌊 Trigger All Workflows",
                  command=self.trigger_all_workflows).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="🛑 Stop All Running",
                  command=self.stop_all_workflows).grid(row=0, column=2)

        # Quick actions
        quick_frame = ttk.LabelFrame(parent, text="⚡ Quick Actions", padding="10")
        quick_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        ttk.Button(quick_frame, text="🔄 Auto-Enhance",
                  command=lambda: self.quick_trigger("auto-enhance")).grid(row=0, column=0, padx=(0, 5), pady=(0, 5))
        ttk.Button(quick_frame, text="🧪 Auto-Test",
                  command=lambda: self.quick_trigger("auto-test")).grid(row=0, column=1, padx=(0, 5), pady=(0, 5))
        ttk.Button(quick_frame, text="📚 Auto-Document",
                  command=lambda: self.quick_trigger("auto-document")).grid(row=1, column=0, padx=(0, 5), pady=(0, 5))
        ttk.Button(quick_frame, text="🚀 Auto-Deploy",
                  command=lambda: self.quick_trigger("auto-deploy")).grid(row=1, column=1, padx=(0, 5), pady=(0, 5))

    def create_output_panel(self, parent):
        """Create the output and history panel"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Execution output tab
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="📤 Execution Output")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            padx=5,
            pady=5
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # History tab
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="📜 Execution History")
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)

        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            padx=5,
            pady=5
        )
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # History controls
        history_controls = ttk.Frame(history_frame)
        history_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(history_controls, text="🔄 Refresh",
                  command=self.refresh_history).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(history_controls, text="🗑️ Clear History",
                  command=self.clear_history).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(history_controls, text="💾 Export History",
                  command=self.export_history).grid(row=0, column=2)

        # System info tab
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="ℹ️ System Info")
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self.info_text = scrolledtext.ScrolledText(
            info_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            padx=5,
            pady=5
        )
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Populate system info
        self.update_system_info()

    def browse_workspace(self):
        """Browse for workspace directory"""
        directory = filedialog.askdirectory(
            title="Select Windsurf Workspace",
            initialdir=self.current_workspace
        )
        if directory:
            self.workspace_var.set(directory)
            self.current_workspace = directory
            self.refresh_workflows()

    def refresh_workflows(self):
        """Refresh the list of available workflows"""
        try:
            self.status_var.set("🔄 Discovering workflows...")

            # Update workspace
            new_workspace = self.workspace_var.get()
            if new_workspace != self.current_workspace:
                self.current_workspace = new_workspace
                self.windsurf_trigger = self._create_windsurf_trigger()

            # Clear current list
            self.workflow_listbox.delete(0, tk.END)

            # Get workflows
            if self.windsurf_trigger:
                workflows = self.windsurf_trigger.list_workflows()

                if workflows:
                    for workflow in workflows:
                        status_icon = "✅" if workflow.get('enabled', True) else "❌"
                        display_text = f"{status_icon} {workflow['name']}"
                        self.workflow_listbox.insert(tk.END, display_text)

                    self.status_var.set(f"✅ Found {len(workflows)} workflows")
                else:
                    self.workflow_listbox.insert(tk.END, "❌ No workflows found")
                    self.status_var.set("❌ No workflows found in workspace")
            else:
                self.workflow_listbox.insert(tk.END, "⚠️ Windsurf trigger not available")
                self.status_var.set("⚠️ Windsurf trigger module not found")

        except Exception as e:
            self.logger.error(f"Failed to refresh workflows: {e}")
            self.status_var.set(f"❌ Error: {str(e)}")
            messagebox.showerror("Refresh Error", f"Failed to refresh workflows:\n{str(e)}")

    def on_workflow_select(self, event=None):
        """Handle workflow selection"""
        selection = self.workflow_listbox.curselection()
        count = len(selection)
        self.selected_count_var.set(f"{count} workflow{'s' if count != 1 else ''}")

    def trigger_selected_workflows(self):
        """Trigger the selected workflows"""
        selection = self.workflow_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select one or more workflows to trigger.")
            return

        # Get selected workflow names
        selected_workflows = []
        for index in selection:
            display_text = self.workflow_listbox.get(index)
            # Extract workflow name (remove status icon)
            workflow_name = display_text.split(' ', 1)[1] if ' ' in display_text else display_text
            selected_workflows.append(workflow_name)

        # Confirm execution
        workflow_list = "\n".join(f"• {w}" for w in selected_workflows)
        mode = "Synchronous" if self.execution_mode.get() == "sync" else "Asynchronous"

        confirm_msg = f"Trigger the following workflows in {mode} mode?\n\n{workflow_list}"

        if not messagebox.askyesno("Confirm Execution", confirm_msg):
            return

        # Execute workflows
        for workflow_name in selected_workflows:
            self._trigger_single_workflow(workflow_name)

    def trigger_all_workflows(self):
        """Trigger all available workflows"""
        if not self.windsurf_trigger:
            messagebox.showerror("Not Available", "Windsurf trigger is not available.")
            return

        workflows = self.windsurf_trigger.list_workflows()
        if not workflows:
            messagebox.showwarning("No Workflows", "No workflows found to trigger.")
            return

        workflow_names = [w['name'] for w in workflows]
        workflow_list = "\n".join(f"• {w}" for w in workflow_names)

        if not messagebox.askyesno("Confirm All",
                                 f"Trigger ALL {len(workflows)} workflows?\n\n{workflow_list}"):
            return

        for workflow_name in workflow_names:
            self._trigger_single_workflow(workflow_name)

    def quick_trigger(self, workflow_name):
        """Quick trigger for common workflows"""
        if not self.windsurf_trigger:
            messagebox.showerror("Not Available", "Windsurf trigger is not available.")
            return

        # Check if workflow exists
        workflows = self.windsurf_trigger.list_workflows()
        if not any(w['name'] == workflow_name for w in workflows):
            messagebox.showwarning("Workflow Not Found",
                                 f"Workflow '{workflow_name}' not found in current workspace.")
            return

        self._trigger_single_workflow(workflow_name)

    def _trigger_single_workflow(self, workflow_name):
        """Trigger a single workflow"""
        if not self.windsurf_trigger:
            return

        # Switch to output tab
        self.notebook.select(0)

        # Add to output
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode = "SYNC" if self.execution_mode.get() == "sync" else "ASYNC"

        self.output_text.insert(tk.END, f"\n[{timestamp}] 🚀 Triggering workflow: {workflow_name} ({mode})\n")
        self.output_text.insert(tk.END, "=" * 60 + "\n")
        self.output_text.see(tk.END)

        # Execute in separate thread
        thread = threading.Thread(
            target=self._execute_workflow_thread,
            args=(workflow_name,),
            daemon=True
        )
        thread.start()

    def _execute_workflow_thread(self, workflow_name):
        """Execute workflow in separate thread"""
        try:
            wait = self.execution_mode.get() == "sync"
            timeout = self.timeout_var.get()

            self.root.after(0, lambda: self.status_var.set(f"⚙️ Executing {workflow_name}..."))

            # Trigger the workflow
            result = self.windsurf_trigger.trigger_workflow(
                workflow_name,
                wait=wait,
                timeout=timeout
            )

            # Update UI with results
            self.root.after(0, lambda: self._display_workflow_result(workflow_name, result))

        except Exception as e:
            error_msg = f"Failed to execute workflow {workflow_name}: {str(e)}"
            self.logger.error(error_msg)
            self.root.after(0, lambda: self._display_workflow_error(workflow_name, error_msg))

    def _display_workflow_result(self, workflow_name, result):
        """Display workflow execution result"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_icon = "✅" if result['success'] else "❌"

        self.output_text.insert(tk.END, f"[{timestamp}] {status_icon} {result['message']}\n")

        if result.get('output'):
            self.output_text.insert(tk.END, f"Output:\n{result['output']}\n")

        if result.get('error'):
            self.output_text.insert(tk.END, f"Errors:\n{result['error']}\n")

        self.output_text.insert(tk.END, "-" * 40 + "\n")
        self.output_text.see(tk.END)

        # Update status
        self.status_var.set("✅ Ready")

        # Add to history
        self._add_to_history(workflow_name, result)

        # Show completion message
        if result['success']:
            messagebox.showinfo("Workflow Complete",
                              f"Workflow '{workflow_name}' completed successfully!")
        else:
            messagebox.showerror("Workflow Failed",
                               f"Workflow '{workflow_name}' failed:\n\n{result['message']}")

    def _display_workflow_error(self, workflow_name, error_msg):
        """Display workflow execution error"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.output_text.insert(tk.END, f"[{timestamp}] ❌ {error_msg}\n")
        self.output_text.insert(tk.END, "-" * 40 + "\n")
        self.output_text.see(tk.END)

        self.status_var.set("❌ Error")

    def stop_all_workflows(self):
        """Stop all running workflows"""
        # This is a placeholder - actual implementation would depend on
        # how Windsurf handles workflow termination
        messagebox.showinfo("Stop Workflows",
                          "Workflow termination not yet implemented.\n\n"
                          "Please wait for running workflows to complete or restart Windsurf.")

    def _add_to_history(self, workflow_name, result):
        """Add execution to history"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'workflow': workflow_name,
            'success': result['success'],
            'message': result['message'],
            'execution_mode': self.execution_mode.get(),
            'timeout': self.timeout_var.get()
        }

        self.workflow_history.insert(0, history_entry)

        # Keep only last 100 entries
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[:100]

        self.save_history()
        self.refresh_history()

    def load_history(self):
        """Load execution history"""
        try:
            if os.path.exists('workflow_panel_history.json'):
                with open('workflow_panel_history.json', 'r') as f:
                    self.workflow_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")
            self.workflow_history = []

    def save_history(self):
        """Save execution history"""
        try:
            with open('workflow_panel_history.json', 'w') as f:
                json.dump(self.workflow_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def refresh_history(self):
        """Refresh the history display"""
        self.history_text.delete(1.0, tk.END)

        if not self.workflow_history:
            self.history_text.insert(tk.END, "No execution history available.\n")
            return

        self.history_text.insert(tk.END, "📜 Workflow Execution History\n")
        self.history_text.insert(tk.END, "=" * 50 + "\n\n")

        for entry in self.workflow_history[:50]:  # Show last 50 entries
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "✅" if entry['success'] else "❌"
            mode = entry.get('execution_mode', 'unknown').upper()

            self.history_text.insert(tk.END, f"[{timestamp}] {status} {entry['workflow']} ({mode})\n")
            self.history_text.insert(tk.END, f"  {entry['message']}\n\n")

    def clear_history(self):
        """Clear execution history"""
        if messagebox.askyesno("Clear History",
                             "Are you sure you want to clear the execution history?"):
            self.workflow_history = []
            self.save_history()
            self.refresh_history()
            messagebox.showinfo("History Cleared", "Execution history has been cleared.")

    def export_history(self):
        """Export execution history to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Workflow History"
            )

            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(self.workflow_history, f, indent=2)
                else:
                    # Export as text
                    with open(filename, 'w') as f:
                        f.write("Workflow Execution History\n")
                        f.write("=" * 50 + "\n\n")
                        for entry in self.workflow_history:
                            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                            status = "SUCCESS" if entry['success'] else "FAILED"
                            f.write(f"[{timestamp}] {status}: {entry['workflow']}\n")
                            f.write(f"  {entry['message']}\n\n")

                messagebox.showinfo("Export Complete",
                                  f"History exported to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export history:\n{str(e)}")

    def update_system_info(self):
        """Update the system information display"""
        self.info_text.delete(1.0, tk.END)

        self.info_text.insert(tk.END, "🖥️ Windsurf Workflow Panel - System Information\n")
        self.info_text.insert(tk.END, "=" * 55 + "\n\n")

        # System info
        self.info_text.insert(tk.END, f"🖥️ Platform: {platform.system()} {platform.release()}\n")
        self.info_text.insert(tk.END, f"🐍 Python: {sys.version}\n")
        self.info_text.insert(tk.END, f"📂 Current Directory: {os.getcwd()}\n")
        self.info_text.insert(tk.END, f"🏠 Workspace: {self.current_workspace}\n\n")

        # Windsurf info
        if self.windsurf_trigger:
            windsurf_path = getattr(self.windsurf_trigger, 'windsurf_path', 'Not found')
            self.info_text.insert(tk.END, f"🎯 Windsurf Executable: {windsurf_path or 'Not found'}\n")
            self.info_text.insert(tk.END, f"📋 Workflows Path: {self.windsurf_trigger.workflows_path}\n\n")
        else:
            self.info_text.insert(tk.END, "⚠️ Windsurf trigger module not available\n\n")

        # Workflow stats
        try:
            workflow_count = self.workflow_listbox.size() if hasattr(self, 'workflow_listbox') else 0
            history_count = len(self.workflow_history)
            self.info_text.insert(tk.END, f"📊 Available Workflows: {workflow_count}\n")
            self.info_text.insert(tk.END, f"📜 History Entries: {history_count}\n\n")
        except:
            pass

        # Features
        self.info_text.insert(tk.END, "✨ Features:\n")
        self.info_text.insert(tk.END, "  • Real-time workflow triggering\n")
        self.info_text.insert(tk.END, "  • Synchronous/Asynchronous execution\n")
        self.info_text.insert(tk.END, "  • Execution history and logging\n")
        self.info_text.insert(tk.END, "  • Quick action buttons\n")
        self.info_text.insert(tk.END, "  • System information display\n")
        self.info_text.insert(tk.END, "  • Export capabilities\n\n")

        # Instructions
        self.info_text.insert(tk.END, "📖 Usage Instructions:\n")
        self.info_text.insert(tk.END, "  1. Select workspace or use auto-detection\n")
        self.info_text.insert(tk.END, "  2. Click 'Refresh Workflows' to load available workflows\n")
        self.info_text.insert(tk.END, "  3. Select workflows and choose execution mode\n")
        self.info_text.insert(tk.END, "  4. Click 'Trigger Selected' or use quick actions\n")
        self.info_text.insert(tk.END, "  5. Monitor execution in the Output tab\n")
        self.info_text.insert(tk.END, "  6. View history in the History tab\n")


def main():
    """Main entry point"""
    app = WindsurfWorkflowPanel()


if __name__ == "__main__":
    main()
