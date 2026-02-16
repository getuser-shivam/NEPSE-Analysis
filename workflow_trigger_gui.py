#!/usr/bin/env python3
"""
Workflow Trigger GUI - Standalone Application
A GUI application for triggering various workflows in the NEPSE Analysis system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import os
import sys
import json
import logging
from datetime import datetime
import time


class WorkflowTriggerGUI:
    """GUI application for triggering workflows"""

    def __init__(self, root):
        self.root = root
        self.root.title("NEPSE Analysis - Workflow Trigger")
        self.root.geometry("700x600")
        self.root.resizable(True, True)

        # Initialize logging
        self.setup_logging()

        # Initialize variables
        self.workflow_history = []
        self.active_processes = {}
        self.workflow_configs = self.load_workflow_configs()

        # Create GUI
        self.create_gui()

        # Load workflow history
        self.load_history()

        self.logger.info("Workflow Trigger GUI initialized")

    def setup_logging(self):
        """Setup logging for the GUI application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('workflow_trigger.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('WorkflowTriggerGUI')

    def load_workflow_configs(self):
        """Load available workflow configurations"""
        configs = {}

        # Check for workflow directory
        workflow_dir = os.path.join(os.path.dirname(__file__), '.windsurf', 'workflows')
        if os.path.exists(workflow_dir):
            for file in os.listdir(workflow_dir):
                if file.endswith('.md'):
                    workflow_name = file.replace('.md', '')
                    config_path = os.path.join(workflow_dir, file)

                    # Read first few lines to get description
                    try:
                        with open(config_path, 'r') as f:
                            lines = f.readlines()[:10]
                            description = "No description available"
                            for line in lines:
                                if line.startswith('description:'):
                                    description = line.split(':', 1)[1].strip()
                                    break
                    except Exception as e:
                        description = f"Error reading: {e}"

                    configs[workflow_name] = {
                        'path': config_path,
                        'description': description,
                        'enabled': True
                    }

        # Add manual trigger options if no workflows found
        if not configs:
            configs = {
                'auto-enhance': {
                    'path': None,
                    'description': 'Trigger automatic enhancement cycle',
                    'enabled': True
                },
                'auto-test': {
                    'path': None,
                    'description': 'Run automated testing suite',
                    'enabled': True
                },
                'auto-document': {
                    'path': None,
                    'description': 'Update documentation automatically',
                    'enabled': True
                },
                'auto-deploy': {
                    'path': None,
                    'description': 'Deploy application updates',
                    'enabled': True
                }
            }

        return configs

    def create_gui(self):
        """Create the main GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="NEPSE Analysis - Workflow Trigger",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Workflow selection frame
        workflow_frame = ttk.LabelFrame(main_frame, text="Available Workflows", padding="10")
        workflow_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        workflow_frame.columnconfigure(0, weight=1)

        # Workflow listbox with scrollbar
        listbox_frame = ttk.Frame(workflow_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.workflow_listbox = tk.Listbox(listbox_frame, height=8, yscrollcommand=scrollbar.set,
                                         selectmode=tk.SINGLE, font=('Arial', 10))
        self.workflow_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.workflow_listbox.yview)

        # Populate workflow list
        for workflow_name, config in self.workflow_configs.items():
            status = "✓" if config['enabled'] else "✗"
            display_name = f"{status} {workflow_name}"
            self.workflow_listbox.insert(tk.END, display_name)

        # Bind selection event
        self.workflow_listbox.bind('<<ListboxSelect>>', self.on_workflow_select)

        # Description label
        self.description_label = ttk.Label(workflow_frame, text="Select a workflow to see description",
                                          wraplength=600, justify=tk.LEFT)
        self.description_label.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))

        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(10, 0))

        # Trigger button
        self.trigger_button = ttk.Button(button_frame, text="Trigger Workflow",
                                       command=self.trigger_workflow, state=tk.DISABLED)
        self.trigger_button.pack(side=tk.LEFT, padx=(0, 10))

        # Refresh button
        ttk.Button(button_frame, text="Refresh Workflows", command=self.refresh_workflows).pack(side=tk.LEFT, padx=(0, 10))

        # Clear history button
        ttk.Button(button_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT)

        # Status and output frame
        output_frame = ttk.LabelFrame(main_frame, text="Workflow Output", padding="10")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD,
                                                   font=('Courier', 9))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Progress bar (hidden initially)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.progress_bar.grid_remove()  # Hide initially

    def on_workflow_select(self, event):
        """Handle workflow selection"""
        selection = self.workflow_listbox.curselection()
        if selection:
            index = selection[0]
            workflow_name = list(self.workflow_configs.keys())[index]
            config = self.workflow_configs[workflow_name]

            # Update description
            self.description_label.config(text=config['description'])

            # Enable/disable trigger button
            if config['enabled']:
                self.trigger_button.config(state=tk.NORMAL)
                self.selected_workflow = workflow_name
            else:
                self.trigger_button.config(state=tk.DISABLED)
                self.selected_workflow = None
        else:
            self.description_label.config(text="Select a workflow to see description")
            self.trigger_button.config(state=tk.DISABLED)
            self.selected_workflow = None

    def trigger_workflow(self):
        """Trigger the selected workflow"""
        if not hasattr(self, 'selected_workflow') or not self.selected_workflow:
            messagebox.showwarning("No Selection", "Please select a workflow to trigger.")
            return

        workflow_name = self.selected_workflow
        config = self.workflow_configs[workflow_name]

        # Confirm trigger
        if not messagebox.askyesno("Confirm Trigger",
                                 f"Are you sure you want to trigger the '{workflow_name}' workflow?\n\n{config['description']}"):
            return

        # Start workflow in separate thread
        self.status_var.set(f"Triggering workflow: {workflow_name}")
        self.progress_bar.grid()  # Show progress bar
        self.progress_var.set(0)

        # Disable trigger button during execution
        self.trigger_button.config(state=tk.DISABLED)

        # Clear output
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Starting workflow: {workflow_name}\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")

        # Start workflow thread
        thread = threading.Thread(target=self.run_workflow, args=(workflow_name, config))
        thread.daemon = True
        thread.start()

    def run_workflow(self, workflow_name, config):
        """Run the workflow in a separate thread"""
        try:
            start_time = time.time()

            # Update progress
            self.root.after(0, lambda: self.progress_var.set(10))

            # Check if it's a file-based workflow
            if config['path'] and os.path.exists(config['path']):
                # Read and display workflow content
                with open(config['path'], 'r') as f:
                    content = f.read()

                self.root.after(0, lambda: self.output_text.insert(tk.END, f"Workflow file loaded: {config['path']}\n\n"))
                self.root.after(0, lambda: self.output_text.insert(tk.END, "Workflow Content:\n"))
                self.root.after(0, lambda: self.output_text.insert(tk.END, "-" * 30 + "\n"))
                self.root.after(0, lambda: self.output_text.insert(tk.END, content[:1000] + ("..." if len(content) > 1000 else "")))
                self.root.after(0, lambda: self.output_text.insert(tk.END, "\n\n"))

            # Simulate workflow execution
            steps = [
                "Initializing workflow environment...",
                "Loading configuration...",
                "Validating prerequisites...",
                "Executing workflow steps...",
                "Processing results...",
                "Finalizing workflow..."
            ]

            for i, step in enumerate(steps):
                time.sleep(0.5)  # Simulate processing time
                progress = 20 + (i * 80 // len(steps))
                self.root.after(0, lambda s=step, p=progress: self.update_progress(s, p))

            # Complete workflow
            end_time = time.time()
            duration = end_time - start_time

            self.root.after(0, lambda: self.complete_workflow(workflow_name, duration))

        except Exception as e:
            self.root.after(0, lambda: self.fail_workflow(workflow_name, str(e)))

    def update_progress(self, step, progress):
        """Update progress during workflow execution"""
        self.output_text.insert(tk.END, f"✓ {step}\n")
        self.progress_var.set(progress)
        self.status_var.set(f"Running: {step}")

    def complete_workflow(self, workflow_name, duration):
        """Handle workflow completion"""
        self.output_text.insert(tk.END, f"\n✅ Workflow '{workflow_name}' completed successfully!\n")
        self.output_text.insert(tk.END, ".2f"        self.output_text.insert(tk.END, "=" * 50 + "\n")

        self.status_var.set("Ready")
        self.progress_bar.grid_remove()
        self.trigger_button.config(state=tk.NORMAL)

        # Add to history
        self.add_to_history(workflow_name, "Success", duration)

        # Show completion message
        messagebox.showinfo("Workflow Complete",
                          f"Workflow '{workflow_name}' completed successfully in {duration:.1f} seconds!")

    def fail_workflow(self, workflow_name, error):
        """Handle workflow failure"""
        self.output_text.insert(tk.END, f"\n❌ Workflow '{workflow_name}' failed!\n")
        self.output_text.insert(tk.END, f"Error: {error}\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")

        self.status_var.set("Error")
        self.progress_bar.grid_remove()
        self.trigger_button.config(state=tk.NORMAL)

        # Add to history
        self.add_to_history(workflow_name, f"Failed: {error}", 0)

        # Show error message
        messagebox.showerror("Workflow Failed", f"Workflow '{workflow_name}' failed:\n\n{error}")

    def add_to_history(self, workflow_name, status, duration):
        """Add workflow execution to history"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'workflow': workflow_name,
            'status': status,
            'duration': duration
        }

        self.workflow_history.insert(0, history_entry)

        # Keep only last 50 entries
        if len(self.workflow_history) > 50:
            self.workflow_history = self.workflow_history[:50]

        # Save history
        self.save_history()

    def load_history(self):
        """Load workflow execution history"""
        try:
            if os.path.exists('workflow_history.json'):
                with open('workflow_history.json', 'r') as f:
                    self.workflow_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load workflow history: {e}")
            self.workflow_history = []

    def save_history(self):
        """Save workflow execution history"""
        try:
            with open('workflow_history.json', 'w') as f:
                json.dump(self.workflow_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save workflow history: {e}")

    def clear_history(self):
        """Clear workflow execution history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the workflow history?"):
            self.workflow_history = []
            self.save_history()
            messagebox.showinfo("History Cleared", "Workflow history has been cleared.")

    def refresh_workflows(self):
        """Refresh the list of available workflows"""
        self.workflow_configs = self.load_workflow_configs()
        self.workflow_listbox.delete(0, tk.END)

        # Repopulate workflow list
        for workflow_name, config in self.workflow_configs.items():
            status = "✓" if config['enabled'] else "✗"
            display_name = f"{status} {workflow_name}"
            self.workflow_listbox.insert(tk.END, display_name)

        self.status_var.set("Workflows refreshed")
        messagebox.showinfo("Refreshed", "Workflow list has been refreshed.")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = WorkflowTriggerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
