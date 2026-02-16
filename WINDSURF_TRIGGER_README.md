# Windsurf Editor Workflow Trigger

A powerful Python application that can programmatically trigger Windsurf editor workflows, manage workspace operations, and automate development tasks.

## Features

### 🖥️ **Editor Integration**
- **Automatic Windsurf Detection**: Finds Windsurf executable across different platforms
- **Workspace Integration**: Works seamlessly with Windsurf workspaces
- **Workflow Discovery**: Automatically discovers all available workflows in `.windsurf/workflows/`
- **Cross-Platform Support**: Windows, macOS, and Linux compatibility

### 🚀 **Workflow Management**
- **Single Workflow Triggering**: Execute specific workflows by name
- **Batch Workflow Execution**: Trigger all available workflows at once
- **Workflow Status Checking**: Verify workflow existence and health
- **Standalone Script Generation**: Create executable scripts for individual workflows

### ⚡ **Execution Modes**
- **Synchronous Execution**: Wait for workflow completion with timeout support
- **Asynchronous Execution**: Fire-and-forget workflow triggering
- **Progress Monitoring**: Real-time execution status and logging
- **Error Handling**: Comprehensive error reporting and recovery

### 📊 **CLI Interface**
- **Rich Command Line**: Full-featured CLI with comprehensive options
- **Workflow Listing**: Browse and inspect available workflows
- **Status Reporting**: Detailed execution results and diagnostics
- **Help System**: Built-in help and usage examples

## Installation

### Prerequisites
- Python 3.6+
- Windsurf Editor installed and accessible
- Access to target workspace

### Quick Install
```bash
# No additional dependencies required (uses only standard library)
# Just ensure the script is executable
chmod +x windsurf_trigger.py  # Linux/Mac only
```

## Usage

### Command Line Interface

```bash
# Basic usage - list all workflows
python windsurf_trigger.py --list

# Trigger a specific workflow
python windsurf_trigger.py --trigger auto-enhance

# Trigger all workflows
python windsurf_trigger.py --trigger-all

# Check workflow status
python windsurf_trigger.py --status auto-test

# Create standalone trigger script
python windsurf_trigger.py --create-script auto-enhance

# Wait for completion with timeout
python windsurf_trigger.py --trigger auto-enhance --wait --timeout 600

# Specify custom workspace
python windsurf_trigger.py --workspace /path/to/project --trigger auto-test

# Verbose logging
python windsurf_trigger.py --trigger auto-enhance --verbose
```

### Windows Batch Launcher
```bash
# Use the batch file for Windows
windsurf_trigger.bat --list
windsurf_trigger.bat --trigger auto-enhance
```

## Commands

### `--list`
List all available workflows in the current workspace.

```bash
python windsurf_trigger.py --list
```
Output:
```
📋 Available workflows in /path/to/workspace:
------------------------------------------------------------
• auto-enhance
  Continuous code improvement and enhancement

• auto-test
  Automated testing suite execution

• auto-document
  Documentation generation and updates
```

### `--trigger WORKFLOW`
Trigger a specific workflow by name.

```bash
python windsurf_trigger.py --trigger auto-enhance
```
Options:
- `--wait`: Wait for completion (default: async)
- `--timeout SECONDS`: Timeout for synchronous execution (default: 300)

### `--trigger-all`
Execute all available workflows in sequence.

```bash
python windsurf_trigger.py --trigger-all --wait
```

### `--status WORKFLOW`
Check the status and information about a specific workflow.

```bash
python windsurf_trigger.py --status auto-test
```
Output:
```
📄 Workflow: auto-test
📍 Path: /workspace/.windsurf/workflows/auto-test.md
📝 Description: Automated testing suite execution
📊 Size: 2048 bytes
🕒 Modified: Mon Feb 16 13:45:30 2026
📖 Readable: Yes
```

### `--create-script WORKFLOW`
Generate a standalone Python script to trigger a specific workflow.

```bash
python windsurf_trigger.py --create-script auto-enhance
```
Creates: `trigger_auto-enhance.py`

## Configuration

### Workspace Detection
The application automatically detects the workspace by looking for:
- `.windsurf/` directory
- `.git/` directory
- Common project files (`main.py`, `requirements.txt`, etc.)

### Windsurf Executable Detection
Searches for Windsurf in standard installation locations:
- **Windows**: `Program Files`, `AppData\Local\Programs`
- **macOS**: `/Applications`, `/usr/local/bin`, `/opt/homebrew/bin`
- **Linux**: `/usr/bin`, `/usr/local/bin`, `/snap/bin`

### Custom Paths
Override automatic detection with command-line options:

```bash
# Specify workspace path
python windsurf_trigger.py --workspace /custom/path

# The executable path is auto-detected but can be overridden
# by modifying the WINDSURF_PATH environment variable
```

## Workflow Integration

### Workflow File Format
Works with standard Windsurf workflow markdown files:

```markdown
---
auto_execution_mode: 1
description: My workflow description
---

# Workflow content here
Workflow instructions and automation steps...
```

### Supported Workflows
Compatible with all Windsurf workflow types:
- **auto-enhance**: Code improvement workflows
- **auto-test**: Testing automation
- **auto-document**: Documentation generation
- **auto-deploy**: Deployment automation
- **Custom workflows**: Any user-defined workflow

## Technical Details

### Architecture
- **Modular Design**: Separate concerns for editor integration, workflow management, and CLI
- **Cross-Platform**: Platform-agnostic executable detection and command building
- **Error Recovery**: Comprehensive error handling with graceful degradation
- **Logging**: Structured logging with configurable verbosity

### Execution Methods
1. **Direct Command**: Attempts to use Windsurf's workflow trigger API
2. **Parameter Passing**: Passes workflow parameters to Windsurf
3. **VSCode Compatibility**: Falls back to VSCode-like command structures
4. **Workspace Opening**: Basic workspace opening as final fallback

### Security Considerations
- **Path Validation**: Validates all file paths before execution
- **Command Sanitization**: Sanitizes workflow names and parameters
- **Timeout Protection**: Prevents hanging processes with configurable timeouts
- **Permission Checking**: Verifies file access permissions

## Troubleshooting

### Common Issues

#### "Windsurf executable not found"
```bash
# Check if Windsurf is installed
which windsurf  # Linux/Mac
where windsurf  # Windows

# Manually specify path
export WINDSURF_PATH=/path/to/windsurf
python windsurf_trigger.py --trigger workflow-name
```

#### "Workflow not found"
```bash
# Check workflow directory exists
ls -la .windsurf/workflows/

# Verify workflow file exists
python windsurf_trigger.py --list
```

#### "Workspace not detected"
```bash
# Specify workspace explicitly
python windsurf_trigger.py --workspace /path/to/project --list

# Check for workspace indicators
ls -la | grep -E "(.windsurf|.git|main.py|requirements.txt)"
```

#### "Permission denied"
```bash
# Check file permissions
ls -la .windsurf/workflows/

# Ensure script is executable
chmod +x windsurf_trigger.py

# Run with appropriate permissions
sudo python windsurf_trigger.py --trigger workflow-name
```

### Debug Mode
Enable verbose logging for troubleshooting:

```bash
python windsurf_trigger.py --trigger workflow-name --verbose
```

Check the generated log file:
```bash
tail -f windsurf_trigger.log
```

### Environment Variables
```bash
# Override Windsurf executable path
export WINDSURF_PATH=/custom/path/to/windsurf

# Override workspace path
export WORKSPACE_PATH=/custom/workspace/path

# Enable debug logging
export WINDSURF_TRIGGER_DEBUG=1
```

## Development

### Extending Functionality
```python
from windsurf_trigger import WindsurfTrigger

# Create custom trigger instance
trigger = WindsurfTrigger(workspace_path="/my/project")

# List workflows
workflows = trigger.list_workflows()

# Trigger workflow programmatically
result = trigger.trigger_workflow("auto-enhance", wait=True)

# Check status
status = trigger.get_workflow_status("auto-test")
```

### Adding New Commands
Modify the `main()` function to add new CLI arguments:

```python
parser.add_argument('--new-command', type=str,
                   help='Description of new command')
```

### Custom Workflow Types
Extend `_extract_workflow_description()` to support new workflow formats.

## Examples

### Automated CI/CD Pipeline
```bash
#!/bin/bash
# Run all quality checks
python windsurf_trigger.py --trigger auto-test --wait
python windsurf_trigger.py --trigger auto-document --wait
python windsurf_trigger.py --trigger auto-deploy --wait
```

### Development Workflow
```python
# Python script for automated development workflow
from windsurf_trigger import WindsurfTrigger

trigger = WindsurfTrigger()

# Run enhancement cycle
result = trigger.trigger_workflow("auto-enhance", wait=True)
if result['success']:
    print("✅ Code enhancement completed")

# Run tests
test_result = trigger.trigger_workflow("auto-test", wait=True)
if test_result['success']:
    print("✅ All tests passed")
```

### Batch Processing
```bash
# Process multiple projects
for project in project1 project2 project3; do
    python windsurf_trigger.py --workspace "/projects/$project" --trigger auto-enhance --wait
done
```

## License

Part of the NEPSE Analysis Tool suite.

## Support

For issues and feature requests:
- Check logs in `windsurf_trigger.log`
- Verify Windsurf installation and paths
- Ensure workflow files are properly formatted
- Test with `--verbose` flag for detailed diagnostics

---

**🎯 Windsurf Editor Workflow Trigger** - Programmatic workflow execution for Windsurf editor

*Automate your development workflows with powerful editor integration!*
