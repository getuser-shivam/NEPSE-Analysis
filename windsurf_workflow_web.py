#!/usr/bin/env python3
"""
Windsurf Workflow Web Interface - Browser-based Workflow Trigger
A web application that provides browser-based access to Windsurf workflow triggering
"""

import os
import sys
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response
from flask_cors import CORS
import webbrowser
import socket
import logging

# Import our existing modules
try:
    import windsurf_trigger
    WINDSURF_TRIGGER_AVAILABLE = True
except ImportError:
    WINDSURF_TRIGGER_AVAILABLE = False
    print("Warning: windsurf_trigger module not available")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global variables
workflow_history = []
active_executions = {}
windsurf_trigger_instance = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Windsurf Workflow Trigger</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .card-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }

        .workflow-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }

        .workflow-card:hover {
            border-color: #007bff;
            background: #e3f2fd;
            transform: translateY(-2px);
        }

        .workflow-card.selected {
            border-color: #28a745;
            background: #d4edda;
        }

        .workflow-name {
            font-weight: bold;
            font-size: 1.1rem;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .workflow-desc {
            color: #6c757d;
            font-size: 0.9rem;
            line-height: 1.4;
        }

        .status-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .status-active {
            background: #28a745;
            color: white;
        }

        .status-inactive {
            background: #dc3545;
            color: white;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }

        .btn-primary {
            background: #007bff;
            color: white;
        }

        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }

        .btn-success {
            background: #28a745;
            color: white;
        }

        .btn-success:hover {
            background: #1e7e34;
        }

        .btn-warning {
            background: #ffc107;
            color: #212529;
        }

        .btn-warning:hover {
            background: #e0a800;
        }

        .btn-danger {
            background: #dc3545;
            color: white;
        }

        .btn-danger:hover {
            background: #c82333;
        }

        .btn-secondary {
            background: #6c757d;
            color: white;
        }

        .btn-secondary:hover {
            background: #5a6268;
        }

        .execution-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .execution-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .execution-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .execution-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .mode-selector {
            display: flex;
            gap: 5px;
        }

        .mode-btn {
            padding: 5px 15px;
            border: 2px solid #007bff;
            background: white;
            color: #007bff;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .mode-btn.active {
            background: #007bff;
            color: white;
        }

        .timeout-input {
            width: 80px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        .output-panel {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .output-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .output-controls {
            display: flex;
            gap: 10px;
        }

        .output-area {
            background: #2d3748;
            color: #e2e8f0;
            border: none;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            width: 100%;
            height: 300px;
            resize: vertical;
            overflow-y: auto;
        }

        .history-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
        }

        .history-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
        }

        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 10px;
            background: #f8f9fa;
        }

        .history-info {
            flex: 1;
        }

        .history-workflow {
            font-weight: bold;
            color: #2c3e50;
        }

        .history-time {
            color: #6c757d;
            font-size: 0.9rem;
        }

        .history-status {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .status-success {
            background: #d4edda;
            color: #155724;
        }

        .status-error {
            background: #f8d7da;
            color: #721c24;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-fill {
            height: 100%;
            background: #007bff;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }

        .stat-label {
            color: #6c757d;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .dashboard {
                grid-template-columns: 1fr;
            }

            .workflow-grid {
                grid-template-columns: 1fr;
            }

            .controls {
                justify-content: center;
            }
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Windsurf Workflow Trigger</h1>
            <p>Browser-based workflow execution interface for Windsurf editor</p>
        </div>

        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value" id="workflowCount">-</div>
                <div class="stat-label">Available Workflows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="historyCount">-</div>
                <div class="stat-label">History Entries</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="activeCount">0</div>
                <div class="stat-label">Active Executions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="successRate">-</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <div class="dashboard">
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📋</span>
                    <span class="card-title">Available Workflows</span>
                </div>
                <div id="workflowGrid" class="workflow-grid">
                    <div class="workflow-card">
                        <div class="workflow-name">Loading...</div>
                        <div class="workflow-desc">Discovering workflows</div>
                    </div>
                </div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="refreshWorkflows()">🔄 Refresh</button>
                    <button class="btn btn-secondary" onclick="selectAll()">☑️ Select All</button>
                    <button class="btn btn-secondary" onclick="clearSelection()">☐ Clear</button>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-icon">⚡</span>
                    <span class="card-title">Quick Actions</span>
                </div>
                <div class="controls">
                    <button class="btn btn-success" onclick="quickTrigger('auto-enhance')">🔄 Auto-Enhance</button>
                    <button class="btn btn-success" onclick="quickTrigger('auto-test')">🧪 Auto-Test</button>
                    <button class="btn btn-success" onclick="quickTrigger('auto-document')">📚 Auto-Document</button>
                    <button class="btn btn-warning" onclick="triggerAll()">🌊 Trigger All</button>
                    <button class="btn btn-danger" onclick="stopAll()">🛑 Stop All</button>
                </div>
            </div>
        </div>

        <div class="execution-panel">
            <div class="execution-header">
                <span class="execution-title">⚙️ Workflow Execution</span>
                <div class="execution-controls">
                    <div class="mode-selector">
                        <button class="mode-btn active" id="asyncBtn" onclick="setMode('async')">⚡ Async</button>
                        <button class="mode-btn" id="syncBtn" onclick="setMode('sync')">⏳ Sync</button>
                    </div>
                    <span>Timeout:</span>
                    <input type="number" class="timeout-input" id="timeoutInput" value="300" min="30" max="3600">
                    <span>sec</span>
                </div>
            </div>
            <div class="controls">
                <button class="btn btn-primary" onclick="triggerSelected()">🚀 Trigger Selected</button>
                <button class="btn btn-secondary" onclick="clearOutput()">🗑️ Clear Output</button>
            </div>
            <div class="progress-bar" id="progressBar" style="display: none;">
                <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
            </div>
        </div>

        <div class="output-panel">
            <div class="output-header">
                <span class="output-title">📤 Execution Output</span>
                <div class="output-controls">
                    <button class="btn btn-secondary" onclick="exportOutput()">💾 Export</button>
                    <button class="btn btn-secondary" onclick="clearOutput()">🗑️ Clear</button>
                </div>
            </div>
            <textarea class="output-area" id="outputArea" readonly>Initializing Windsurf Workflow Trigger...</textarea>
        </div>

        <div class="history-panel">
            <div class="history-title">📜 Execution History</div>
            <div id="historyContainer">
                <div class="history-item">
                    <div class="history-info">
                        <div class="history-workflow">System initialized</div>
                        <div class="history-time">Just now</div>
                    </div>
                    <span class="history-status status-success">READY</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedWorkflows = [];
        let currentMode = 'async';
        let executionTimeout = 300;
        let availableWorkflows = [];

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            refreshWorkflows();
            updateStats();
            logOutput('Windsurf Workflow Web Interface initialized');
        });

        // Workflow management
        async function refreshWorkflows() {
            try {
                logOutput('🔄 Refreshing workflows...');
                const response = await fetch('/api/workflows');
                const workflows = await response.json();

                availableWorkflows = workflows;
                renderWorkflows(workflows);
                updateStats();
                logOutput(`✅ Found ${workflows.length} workflows`);
            } catch (error) {
                logOutput(`❌ Failed to refresh workflows: ${error.message}`);
            }
        }

        function renderWorkflows(workflows) {
            const grid = document.getElementById('workflowGrid');
            grid.innerHTML = '';

            workflows.forEach(workflow => {
                const card = document.createElement('div');
                card.className = 'workflow-card';
                card.onclick = () => toggleWorkflow(workflow.name);

                const statusClass = 'status-active';
                const statusText = 'READY';

                card.innerHTML = `
                    <div class="workflow-name">${workflow.name}</div>
                    <div class="workflow-desc">${workflow.description || 'No description available'}</div>
                    <span class="status-badge ${statusClass}">${statusText}</span>
                `;

                grid.appendChild(card);
            });
        }

        function toggleWorkflow(workflowName) {
            const index = selectedWorkflows.indexOf(workflowName);
            if (index > -1) {
                selectedWorkflows.splice(index, 1);
            } else {
                selectedWorkflows.push(workflowName);
            }
            updateWorkflowSelection();
        }

        function updateWorkflowSelection() {
            document.querySelectorAll('.workflow-card').forEach(card => {
                const workflowName = card.querySelector('.workflow-name').textContent;
                if (selectedWorkflows.includes(workflowName)) {
                    card.classList.add('selected');
                } else {
                    card.classList.remove('selected');
                }
            });
        }

        function selectAll() {
            selectedWorkflows = availableWorkflows.map(w => w.name);
            updateWorkflowSelection();
        }

        function clearSelection() {
            selectedWorkflows = [];
            updateWorkflowSelection();
        }

        // Execution controls
        function setMode(mode) {
            currentMode = mode;
            document.getElementById('asyncBtn').classList.toggle('active', mode === 'async');
            document.getElementById('syncBtn').classList.toggle('active', mode === 'sync');
        }

        async function triggerSelected() {
            if (selectedWorkflows.length === 0) {
                alert('Please select at least one workflow to trigger.');
                return;
            }

            const confirmed = confirm(`Trigger ${selectedWorkflows.length} workflow(s) in ${currentMode.toUpperCase()} mode?`);
            if (!confirmed) return;

            executionTimeout = parseInt(document.getElementById('timeoutInput').value) || 300;

            showProgress();
            logOutput(`🚀 Triggering ${selectedWorkflows.length} workflows...`);

            try {
                const response = await fetch('/api/trigger', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        workflows: selectedWorkflows,
                        mode: currentMode,
                        timeout: executionTimeout
                    })
                });

                const result = await response.json();

                if (result.success) {
                    logOutput(`✅ Execution started successfully`);
                    result.results.forEach(r => {
                        logOutput(`${r.success ? '✅' : '❌'} ${r.workflow}: ${r.message}`);
                    });
                } else {
                    logOutput(`❌ Execution failed: ${result.message}`);
                }

            } catch (error) {
                logOutput(`❌ Trigger failed: ${error.message}`);
            }

            hideProgress();
            refreshHistory();
        }

        async function triggerAll() {
            selectedWorkflows = availableWorkflows.map(w => w.name);
            updateWorkflowSelection();
            await triggerSelected();
        }

        async function quickTrigger(workflowName) {
            selectedWorkflows = [workflowName];
            updateWorkflowSelection();
            await triggerSelected();
        }

        function stopAll() {
            // Implementation for stopping workflows would go here
            logOutput('🛑 Stop functionality not yet implemented');
        }

        // Progress management
        function showProgress() {
            document.getElementById('progressBar').style.display = 'block';
            updateProgress(0);
        }

        function hideProgress() {
            document.getElementById('progressBar').style.display = 'none';
        }

        function updateProgress(percent) {
            document.getElementById('progressFill').style.width = `${percent}%`;
        }

        // Output management
        function logOutput(message) {
            const timestamp = new Date().toLocaleTimeString();
            const outputArea = document.getElementById('outputArea');
            outputArea.value += `[${timestamp}] ${message}\n`;
            outputArea.scrollTop = outputArea.scrollHeight;
        }

        function clearOutput() {
            document.getElementById('outputArea').value = '';
        }

        function exportOutput() {
            const output = document.getElementById('outputArea').value;
            const blob = new Blob([output], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `workflow_output_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // History management
        async function refreshHistory() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();

                renderHistory(history);
                updateStats();
            } catch (error) {
                logOutput(`Failed to refresh history: ${error.message}`);
            }
        }

        function renderHistory(history) {
            const container = document.getElementById('historyContainer');
            container.innerHTML = '';

            history.slice(0, 10).forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'history-item';

                const timestamp = new Date(item.timestamp).toLocaleString();
                const statusClass = item.success ? 'status-success' : 'status-error';
                const statusText = item.success ? 'SUCCESS' : 'FAILED';

                itemDiv.innerHTML = `
                    <div class="history-info">
                        <div class="history-workflow">${item.workflow}</div>
                        <div class="history-time">${timestamp}</div>
                    </div>
                    <span class="history-status ${statusClass}">${statusText}</span>
                `;

                container.appendChild(itemDiv);
            });
        }

        // Stats management
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();

                document.getElementById('workflowCount').textContent = stats.workflows || 0;
                document.getElementById('historyCount').textContent = stats.history || 0;
                document.getElementById('activeCount').textContent = stats.active || 0;
                document.getElementById('successRate').textContent = `${stats.success_rate || 0}%`;
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }

        // Periodic updates
        setInterval(() => {
            updateStats();
        }, 5000);
    </script>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/workflows')
def get_workflows():
    """Get available workflows"""
    if not WINDSURF_TRIGGER_AVAILABLE:
        return jsonify([])

    try:
        global windsurf_trigger_instance
        if windsurf_trigger_instance is None:
            windsurf_trigger_instance = windsurf_trigger.WindsurfTrigger()

        workflows = windsurf_trigger_instance.list_workflows()
        return jsonify(workflows)
    except Exception as e:
        app.logger.error(f"Failed to get workflows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger', methods=['POST'])
def trigger_workflows():
    """Trigger workflows"""
    if not WINDSURF_TRIGGER_AVAILABLE:
        return jsonify({'success': False, 'message': 'Windsurf trigger not available'})

    try:
        data = request.get_json()
        workflows = data.get('workflows', [])
        mode = data.get('mode', 'async')
        timeout = data.get('timeout', 300)

        results = []
        for workflow in workflows:
            result = windsurf_trigger_instance.trigger_workflow(
                workflow,
                wait=(mode == 'sync'),
                timeout=timeout
            )
            results.append(result)

            # Add to history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'workflow': workflow,
                'success': result['success'],
                'message': result['message'],
                'mode': mode,
                'timeout': timeout
            }
            workflow_history.insert(0, history_entry)

            # Keep only last 100 entries
            if len(workflow_history) > 100:
                workflow_history[:] = workflow_history[:100]

        return jsonify({
            'success': True,
            'results': results,
            'message': f'Triggered {len(workflows)} workflows'
        })

    except Exception as e:
        app.logger.error(f"Failed to trigger workflows: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Get execution history"""
    return jsonify(workflow_history)

@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    total_workflows = len(windsurf_trigger_instance.list_workflows()) if windsurf_trigger_instance else 0
    total_history = len(workflow_history)
    successful_executions = sum(1 for h in workflow_history if h['success'])
    success_rate = int((successful_executions / total_history * 100) if total_history > 0 else 0)

    return jsonify({
        'workflows': total_workflows,
        'history': total_history,
        'active': len(active_executions),
        'success_rate': success_rate
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'windsurf_trigger': WINDSURF_TRIGGER_AVAILABLE
    })


class WindsurfWorkflowWebApp:
    """Web application for Windsurf workflow triggering"""

    def __init__(self, host='localhost', port=5000, auto_open=True):
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self.server_thread = None
        self.logger = logging.getLogger('WindsurfWorkflowWebApp')

    def start(self):
        """Start the web application"""
        self.logger.info(f"Starting Windsurf Workflow Web App on {self.host}:{self.port}")

        # Start server in separate thread
        self.server_thread = threading.Thread(
            target=lambda: app.run(host=self.host, port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        self.server_thread.start()

        # Wait a moment for server to start
        time.sleep(1)

        # Open browser
        if self.auto_open:
            try:
                url = f"http://{self.host}:{self.port}"
                webbrowser.open(url)
                self.logger.info(f"Opened browser to {url}")
            except Exception as e:
                self.logger.error(f"Failed to open browser: {e}")

        self.logger.info("Windsurf Workflow Web App started successfully")

    def stop(self):
        """Stop the web application"""
        # Flask doesn't provide a clean way to stop, but since it's daemon thread, it will stop with main process
        self.logger.info("Stopping Windsurf Workflow Web App")

    def is_running(self):
        """Check if server is running"""
        return self.server_thread and self.server_thread.is_alive()


def find_free_port():
    """Find a free port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Windsurf Workflow Web Interface - Browser-based workflow triggering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start on localhost:5000
  %(prog)s --port 8080             # Start on custom port
  %(prog)s --host 0.0.0.0         # Bind to all interfaces
  %(prog)s --no-browser           # Don't open browser automatically
        """
    )

    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int,
                       help='Port to run on (default: auto-find free port)')
    parser.add_argument('--no-browser', action='store_true',
                       help="Don't automatically open browser")

    args = parser.parse_args()

    # Find free port if not specified
    if not args.port:
        args.port = find_free_port()

    print(f"🚀 Starting Windsurf Workflow Web Interface...")
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"🔧 Windsurf Trigger Available: {WINDSURF_TRIGGER_AVAILABLE}")
    print(f"📊 Press Ctrl+C to stop")
    print("-" * 50)

    # Create and start web app
    web_app = WindsurfWorkflowWebApp(
        host=args.host,
        port=args.port,
        auto_open=not args.no_browser
    )

    try:
        web_app.start()

        # Keep running
        while web_app.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        web_app.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        web_app.stop()


if __name__ == "__main__":
    main()
