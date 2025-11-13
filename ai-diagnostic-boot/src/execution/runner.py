"""
Script execution runner.
Executes fix scripts with monitoring, timeout, and error handling.
"""

import subprocess
import time
import signal
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExecutionResult:
    """Represents script execution result."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timestamp: str
    error_message: Optional[str] = None
    timed_out: bool = False


class ScriptRunner:
    """Executes scripts with safety controls."""

    def __init__(self, timeout: int = 300,
                 max_output_size: int = 1024 * 1024):  # 1MB
        """
        Initialize script runner.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in bytes
        """
        self.timeout = timeout
        self.max_output_size = max_output_size

    def execute_script(self, script_path: str,
                      args: list = None,
                      dry_run: bool = False) -> ExecutionResult:
        """
        Execute a script file.

        Args:
            script_path: Path to script file
            args: Additional arguments
            dry_run: If True, only show what would be executed

        Returns:
            ExecutionResult object
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        if dry_run:
            # Dry run mode - just validate script exists and is executable
            if not os.path.exists(script_path):
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Script not found: {script_path}",
                    duration=0,
                    timestamp=timestamp,
                    error_message="Script file not found"
                )

            return ExecutionResult(
                success=True,
                exit_code=0,
                stdout=f"[DRY RUN] Would execute: {script_path}",
                stderr="",
                duration=time.time() - start_time,
                timestamp=timestamp
            )

        # Build command
        command = [script_path]
        if args:
            command.extend(args)

        try:
            # Execute script with timeout
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
                timed_out = False

            except subprocess.TimeoutExpired:
                # Kill process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)

                # Force kill if still running
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                stdout, stderr = process.communicate()
                exit_code = -1
                timed_out = True

            duration = time.time() - start_time

            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

            success = (exit_code == 0) and not timed_out

            error_message = None
            if timed_out:
                error_message = f"Script execution timed out after {self.timeout} seconds"
            elif exit_code != 0:
                error_message = f"Script exited with code {exit_code}"

            return ExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                timestamp=timestamp,
                error_message=error_message,
                timed_out=timed_out
            )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Script not found or not executable: {script_path}",
                duration=time.time() - start_time,
                timestamp=timestamp,
                error_message="File not found"
            )

        except PermissionError:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Permission denied: {script_path}",
                duration=time.time() - start_time,
                timestamp=timestamp,
                error_message="Permission denied"
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
                timestamp=timestamp,
                error_message=f"Execution error: {str(e)}"
            )

    def execute_command(self, command: str,
                       shell: bool = True,
                       dry_run: bool = False) -> ExecutionResult:
        """
        Execute a shell command.

        Args:
            command: Command string to execute
            shell: Whether to use shell execution
            dry_run: If True, only show what would be executed

        Returns:
            ExecutionResult object
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        if dry_run:
            return ExecutionResult(
                success=True,
                exit_code=0,
                stdout=f"[DRY RUN] Would execute: {command}",
                stderr="",
                duration=time.time() - start_time,
                timestamp=timestamp
            )

        try:
            process = subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
                timed_out = False

            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)

                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                stdout, stderr = process.communicate()
                exit_code = -1
                timed_out = True

            duration = time.time() - start_time

            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

            success = (exit_code == 0) and not timed_out

            error_message = None
            if timed_out:
                error_message = f"Command timed out after {self.timeout} seconds"
            elif exit_code != 0:
                error_message = f"Command exited with code {exit_code}"

            return ExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                timestamp=timestamp,
                error_message=error_message,
                timed_out=timed_out
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
                timestamp=timestamp,
                error_message=f"Execution error: {str(e)}"
            )

    def execute_with_retry(self, script_path: str,
                          max_retries: int = 3,
                          retry_delay: int = 2) -> ExecutionResult:
        """
        Execute script with automatic retry on failure.

        Args:
            script_path: Path to script
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            ExecutionResult from last attempt
        """
        last_result = None

        for attempt in range(max_retries):
            result = self.execute_script(script_path)

            if result.success:
                return result

            last_result = result

            # Don't retry on timeout or file not found
            if result.timed_out or result.error_message == "File not found":
                break

            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        return last_result

    def verify_execution(self, result: ExecutionResult,
                        expected_patterns: list = None) -> Tuple[bool, str]:
        """
        Verify execution result against expected patterns.

        Args:
            result: ExecutionResult to verify
            expected_patterns: List of patterns to search for in output

        Returns:
            Tuple of (verification_passed, message)
        """
        if not result.success:
            return False, f"Execution failed: {result.error_message}"

        if expected_patterns:
            for pattern in expected_patterns:
                if pattern not in result.stdout and pattern not in result.stderr:
                    return False, f"Expected pattern not found: {pattern}"

        return True, "Execution verified successfully"

    def get_execution_summary(self, results: list) -> Dict:
        """
        Generate summary from multiple execution results.

        Args:
            results: List of ExecutionResult objects

        Returns:
            Summary dictionary
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        timed_out = sum(1 for r in results if r.timed_out)
        total_duration = sum(r.duration for r in results)

        return {
            'total_executions': total,
            'successful': successful,
            'failed': failed,
            'timed_out': timed_out,
            'total_duration': round(total_duration, 2),
            'average_duration': round(total_duration / total if total > 0 else 0, 2),
            'success_rate': round((successful / total * 100) if total > 0 else 0, 1)
        }
