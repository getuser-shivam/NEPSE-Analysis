# Workflow Trigger GUI

A standalone GUI application for triggering and managing workflows in the NEPSE Analysis system.

## Features

- **Workflow Discovery**: Automatically discovers available workflows from the `.windsurf/workflows` directory
- **Interactive GUI**: User-friendly interface for selecting and triggering workflows
- **Real-time Output**: Live display of workflow execution progress and results
- **Execution History**: Tracks all workflow executions with timestamps and status
- **Progress Monitoring**: Visual progress bars for long-running workflows
- **Error Handling**: Comprehensive error reporting and status updates

## Quick Start

### Windows
```bash
# Double-click the batch file
run_workflow_gui.bat
```

### Linux/Mac
```bash
python workflow_trigger_gui.py
```

### Manual Installation
```bash
# Install dependencies
pip install tk

# Run the application
python workflow_trigger_gui.py
```

## Usage

1. **Launch the Application**: Run `python workflow_trigger_gui.py` or `run_workflow_gui.bat`

2. **Select a Workflow**: Click on any available workflow from the list

3. **Review Description**: Read the workflow description in the info panel

4. **Trigger Execution**: Click "Trigger Workflow" to start the selected workflow

5. **Monitor Progress**: Watch the real-time output and progress bar

6. **View Results**: Check the output area for completion status and results

## Available Workflows

The GUI automatically discovers workflows from `.windsurf/workflows/`:

- **auto-enhance**: Continuous code improvement and enhancement
- **auto-test**: Automated testing suite execution
- **auto-document**: Documentation generation and updates
- **auto-deploy**: Application deployment workflows
- **auto-continuous**: Never-ending enhancement cycles
- **review**: Code review and quality checks

## Features Overview

### Workflow Management
- **Automatic Discovery**: Scans for available workflow files
- **Status Indicators**: Shows enabled/disabled workflow status
- **Description Display**: Shows workflow purpose and functionality
- **Validation**: Ensures workflows are available before triggering

### Execution Monitoring
- **Live Output**: Real-time display of workflow execution
- **Progress Tracking**: Visual progress bars for long operations
- **Status Updates**: Current operation status in status bar
- **Error Reporting**: Detailed error messages and troubleshooting

### History Management
- **Execution History**: Tracks all workflow runs with timestamps
- **Success/Failure Status**: Records outcome of each execution
- **Duration Tracking**: Measures execution time for performance analysis
- **Persistent Storage**: Saves history to JSON file for future reference

### User Interface
- **Modern Design**: Clean tkinter interface with proper styling
- **Responsive Layout**: Adapts to different window sizes
- **Intuitive Controls**: Easy-to-use buttons and selection lists
- **Context Help**: Tooltips and descriptions for all features

## Technical Details

### Dependencies
- Python 3.6+
- tkinter (included with Python)
- json (standard library)
- threading (standard library)
- subprocess (standard library)

### File Structure
```
NEPSE-Analysis/
├── workflow_trigger_gui.py    # Main GUI application
├── run_workflow_gui.bat      # Windows launcher
├── workflow_history.json      # Execution history (auto-generated)
└── workflow_trigger.log       # Application logs (auto-generated)
```

### Configuration
The application automatically:
- Discovers workflows from `.windsurf/workflows/`
- Loads execution history from `workflow_history.json`
- Saves logs to `workflow_trigger.log`
- Maintains user preferences and settings

## Troubleshooting

### Common Issues

#### GUI Won't Start
```bash
# Check Python installation
python --version

# Install tkinter if missing (Linux)
sudo apt-get install python3-tk

# Run with verbose output
python -c "import tkinter; print('tkinter OK')"
```

#### Workflows Not Found
```bash
# Check workflow directory exists
ls -la .windsurf/workflows/

# Ensure workflow files have .md extension
find .windsurf/workflows/ -name "*.md"
```

#### Permission Errors
```bash
# Check write permissions for history file
touch workflow_history.json

# Run with appropriate permissions
python workflow_trigger_gui.py
```

### Log Files
Check `workflow_trigger.log` for detailed error information:
```bash
tail -f workflow_trigger.log
```

### Reset Application
To reset the application state:
```bash
# Remove history and logs
rm workflow_history.json workflow_trigger.log

# Restart application
python workflow_trigger_gui.py
```

## Development

### Adding New Workflows
1. Create new workflow file in `.windsurf/workflows/your_workflow.md`
2. Add frontmatter with description:
```markdown
---
description: Your workflow description here
---
Workflow content here...
```
3. Restart the GUI application
4. New workflow will appear in the list

### Extending Functionality
The application is designed to be extensible:

- **Workflow Discovery**: Modify `load_workflow_configs()` for custom discovery
- **Execution Logic**: Extend `run_workflow()` for custom execution methods
- **UI Components**: Add new widgets in `create_gui()` method
- **History Management**: Enhance `add_to_history()` for additional tracking

## License

This application is part of the NEPSE Analysis Tool suite.

## Support

For issues and feature requests:
- Check the logs in `workflow_trigger.log`
- Review workflow execution history
- Ensure workflow files are properly formatted
- Verify file permissions and Python environment
