#!/usr/bin/env python3
"""
Windsurf Editor Trigger - Programmatic Workflow Execution
A Python application that can trigger Windsurf editor workflows programmatically
"""

import subprocess
import sys
import os
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform


class WindsurfTrigger:
    """Programmatic trigger for Windsurf editor workflows"""

    def __init__(self, workspace_path: str = None, log_level: str = "INFO"):
        self.workspace_path = workspace_path or self._find_workspace()
        self.logger = self._setup_logging(log_level)
        self.windsurf_path = self._find_windsurf_executable()
        self.workflows_path = self._get_workflows_path()

        self.logger.info(f"WindsurfTrigger initialized for workspace: {self.workspace_path}")
        self.logger.info(f"Windsurf executable: {self.windsurf_path}")
        self.logger.info(f"Workflows path: {self.workflows_path}")

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('windsurf_trigger.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('WindsurfTrigger')

    def _find_workspace(self) -> str:
        """Find the workspace path"""
        # Try to find workspace from current directory or parent directories
        current_path = Path.cwd()

        # Look for common workspace indicators
        workspace_indicators = ['.windsurf', '.git', 'main.py', 'requirements.txt']

        for path in [current_path] + list(current_path.parents):
            if any((path / indicator).exists() for indicator in workspace_indicators):
                return str(path)

        # Default to current directory
        return str(current_path)

    def _find_windsurf_executable(self) -> Optional[str]:
        """Find the Windsurf executable path"""
        system = platform.system().lower()

        possible_paths = []

        if system == "windows":
            # Windows paths
            possible_paths.extend([
                "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\windsurf\\Windsurf.exe",
                "C:\\Program Files\\windsurf\\Windsurf.exe",
                "C:\\Program Files (x86)\\windsurf\\Windsurf.exe",
                "%LOCALAPPDATA%\\Programs\\windsurf\\Windsurf.exe"
            ])

            # Try to find in PATH
            try:
                result = subprocess.run(['where', 'windsurf'], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            except Exception:
                pass

        elif system == "darwin":  # macOS
            possible_paths.extend([
                "/Applications/Windsurf.app/Contents/MacOS/Windsurf",
                "/usr/local/bin/windsurf",
                "/opt/homebrew/bin/windsurf"
            ])

        elif system == "linux":
            possible_paths.extend([
                "/usr/bin/windsurf",
                "/usr/local/bin/windsurf",
                "/opt/windsurf/windsurf",
                "/snap/bin/windsurf"
            ])

        # Try common paths
        for path in possible_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path

        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'windsurf'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        self.logger.warning("Windsurf executable not found in common locations")
        return None

    def _get_workflows_path(self) -> str:
        """Get the workflows directory path"""
        return os.path.join(self.workspace_path, '.windsurf', 'workflows')

    def list_workflows(self) -> List[Dict[str, str]]:
        """List all available workflows"""
        workflows = []

        if not os.path.exists(self.workflows_path):
            self.logger.warning(f"Workflows directory not found: {self.workflows_path}")
            return workflows

        for file_path in Path(self.workflows_path).glob('*.md'):
            try:
                workflow_name = file_path.stem
                description = self._extract_workflow_description(file_path)

                workflows.append({
                    'name': workflow_name,
                    'path': str(file_path),
                    'description': description
                })

            except Exception as e:
                self.logger.error(f"Error reading workflow {file_path}: {e}")

        return workflows

    def _extract_workflow_description(self, workflow_path: Path) -> str:
        """Extract workflow description from frontmatter"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for description in frontmatter
            if '---' in content:
                parts = content.split('---', 2)
                if len(parts) >= 2:
                    frontmatter = parts[1]
                    for line in frontmatter.split('\n'):
                        line = line.strip()
                        if line.startswith('description:'):
                            return line.split(':', 1)[1].strip()

            # Fallback: use first line or filename
            lines = content.split('\n', 10)
            for line in lines[:5]:
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 10:
                    return line[:100] + '...' if len(line) > 100 else line

            return f"Workflow: {workflow_path.stem}"

        except Exception as e:
            return f"Error reading description: {e}"

    def trigger_workflow(self, workflow_name: str, wait: bool = False, timeout: int = 300) -> Dict[str, Any]:
        """Trigger a specific workflow in Windsurf"""
        result = {
            'success': False,
            'workflow': workflow_name,
            'message': '',
            'return_code': None,
            'output': '',
            'error': ''
        }

        try:
            # Check if workflow exists
            workflows = self.list_workflows()
            workflow = next((w for w in workflows if w['name'] == workflow_name), None)

            if not workflow:
                result['message'] = f"Workflow '{workflow_name}' not found"
                self.logger.error(result['message'])
                return result

            # Build command to trigger workflow
            cmd = self._build_trigger_command(workflow_name)

            if not cmd:
                result['message'] = "Unable to build trigger command for Windsurf"
                self.logger.error(result['message'])
                return result

            self.logger.info(f"Triggering workflow: {workflow_name}")
            self.logger.debug(f"Command: {' '.join(cmd)}")

            # Execute command
            if wait:
                # Run synchronously and wait
                process = subprocess.run(
                    cmd,
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                result['return_code'] = process.returncode
                result['output'] = process.stdout
                result['error'] = process.stderr
                result['success'] = process.returncode == 0

                if result['success']:
                    result['message'] = f"Workflow '{workflow_name}' completed successfully"
                else:
                    result['message'] = f"Workflow '{workflow_name}' failed with code {process.returncode}"

            else:
                # Run asynchronously
                process = subprocess.Popen(
                    cmd,
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                result['message'] = f"Workflow '{workflow_name}' started asynchronously (PID: {process.pid})"
                result['success'] = True

            self.logger.info(result['message'])

        except subprocess.TimeoutExpired:
            result['message'] = f"Workflow '{workflow_name}' timed out after {timeout} seconds"
            self.logger.error(result['message'])
        except Exception as e:
            result['message'] = f"Error triggering workflow '{workflow_name}': {str(e)}"
            self.logger.error(result['message'])

        return result

    def _build_trigger_command(self, workflow_name: str) -> Optional[List[str]]:
        """Build the command to trigger a workflow in Windsurf"""
        if not self.windsurf_path:
            return None

        # Try different approaches based on Windsurf's capabilities

        # Method 1: Direct workflow trigger (if supported)
        cmd = [
            self.windsurf_path,
            '--trigger-workflow', workflow_name,
            '--workspace', self.workspace_path
        ]

        # Method 2: Open with workflow parameter
        if not self._test_command(cmd):
            cmd = [
                self.windsurf_path,
                self.workspace_path,
                '--workflow', workflow_name
            ]

        # Method 3: Use VSCode-like commands (Windsurf might support these)
        if not self._test_command(cmd):
            cmd = [
                self.windsurf_path,
                '--command', f'workbench.action.openWorkflow',
                '--args', json.dumps({'workflow': workflow_name})
            ]

        # Method 4: Simple workspace open (fallback)
        if not self._test_command(cmd):
            cmd = [
                self.windsurf_path,
                self.workspace_path
            ]

        return cmd

    def _test_command(self, cmd: List[str]) -> bool:
        """Test if a command structure is valid (without executing)"""
        try:
            # Just check if the executable exists and command structure looks valid
            if not cmd or not os.path.exists(cmd[0]):
                return False
            return True
        except Exception:
            return False

    def trigger_all_workflows(self, wait: bool = False) -> List[Dict[str, Any]]:
        """Trigger all available workflows"""
        workflows = self.list_workflows()
        results = []

        for workflow in workflows:
            result = self.trigger_workflow(workflow['name'], wait=wait)
            results.append(result)

            # Small delay between triggers
            time.sleep(1)

        return results

    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get the status of a workflow"""
        try:
            workflows = self.list_workflows()
            workflow = next((w for w in workflows if w['name'] == workflow_name), None)

            if not workflow:
                return {'found': False, 'message': f"Workflow '{workflow_name}' not found"}

            # Check if workflow file exists and is readable
            path = Path(workflow['path'])
            stats = path.stat()

            return {
                'found': True,
                'name': workflow_name,
                'path': workflow['path'],
                'description': workflow['description'],
                'size': stats.st_size,
                'modified': time.ctime(stats.st_mtime),
                'readable': os.access(workflow['path'], os.R_OK)
            }

        except Exception as e:
            return {'found': False, 'message': f"Error checking workflow status: {e}"}

    def create_workflow_trigger_script(self, workflow_name: str, output_path: Optional[str] = None) -> str:
        """Create a standalone script to trigger a specific workflow"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script to trigger '{workflow_name}' workflow
Generated by WindsurfTrigger
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from windsurf_trigger import WindsurfTrigger

def main():
    trigger = WindsurfTrigger()
    result = trigger.trigger_workflow("{workflow_name}", wait=True)

    if result['success']:
        print(f"✅ Workflow '{workflow_name}' completed successfully")
        return 0
    else:
        print(f"❌ Workflow '{workflow_name}' failed: {{result['message']}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

        if not output_path:
            output_path = f"trigger_{workflow_name}.py"

        with open(output_path, 'w') as f:
            f.write(script_content)

        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod(output_path, 0o755)

        return output_path


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Windsurf Editor Workflow Trigger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                          # List all workflows
  %(prog)s --trigger auto-enhance          # Trigger specific workflow
  %(prog)s --trigger-all                   # Trigger all workflows
  %(prog)s --status auto-test              # Check workflow status
  %(prog)s --create-script auto-enhance    # Create standalone trigger script
  %(prog)s --workspace /path/to/project    # Specify workspace path
        """
    )

    parser.add_argument('--list', action='store_true',
                       help='List all available workflows')
    parser.add_argument('--trigger', type=str, metavar='WORKFLOW',
                       help='Trigger a specific workflow')
    parser.add_argument('--trigger-all', action='store_true',
                       help='Trigger all available workflows')
    parser.add_argument('--status', type=str, metavar='WORKFLOW',
                       help='Check status of a specific workflow')
    parser.add_argument('--create-script', type=str, metavar='WORKFLOW',
                       help='Create a standalone script to trigger the workflow')
    parser.add_argument('--workspace', type=str,
                       help='Specify workspace path (default: auto-detect)')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for workflow completion (default: async)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for workflow execution in seconds (default: 300)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"

    try:
        trigger = WindsurfTrigger(
            workspace_path=args.workspace,
            log_level=log_level
        )

        if args.list:
            workflows = trigger.list_workflows()
            if workflows:
                print(f"📋 Available workflows in {trigger.workspace_path}:")
                print("-" * 60)
                for workflow in workflows:
                    print(f"• {workflow['name']}")
                    print(f"  {workflow['description']}")
                    print()
            else:
                print("❌ No workflows found in workspace")

        elif args.trigger:
            result = trigger.trigger_workflow(args.trigger, wait=args.wait, timeout=args.timeout)
            if result['success']:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['message']}")
                if result['error']:
                    print(f"Error details: {result['error']}")
                sys.exit(1)

        elif args.trigger_all:
            results = trigger.trigger_all_workflows(wait=args.wait)
            success_count = sum(1 for r in results if r['success'])
            print(f"📊 Triggered {len(results)} workflows, {success_count} successful")

            for result in results:
                status = "✅" if result['success'] else "❌"
                print(f"{status} {result['workflow']}: {result['message']}")

        elif args.status:
            status = trigger.get_workflow_status(args.status)
            if status['found']:
                print(f"📄 Workflow: {status['name']}")
                print(f"📍 Path: {status['path']}")
                print(f"📝 Description: {status['description']}")
                print(f"📊 Size: {status['size']} bytes")
                print(f"🕒 Modified: {status['modified']}")
                print(f"📖 Readable: {'Yes' if status['readable'] else 'No'}")
            else:
                print(f"❌ {status['message']}")
                sys.exit(1)

        elif args.create_script:
            script_path = trigger.create_workflow_trigger_script(args.create_script)
            print(f"📝 Created trigger script: {script_path}")
            print(f"💡 Run with: python {script_path}")

        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
