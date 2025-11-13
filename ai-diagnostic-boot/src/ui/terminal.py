"""
Terminal UI for AI Diagnostic Boot Drive.
Provides text-based user interface with colors and formatting.
"""

import sys
import time
from typing import List, Dict, Optional


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


class TerminalUI:
    """Terminal-based user interface."""

    def __init__(self, color_enabled: bool = True):
        """
        Initialize terminal UI.

        Args:
            color_enabled: Whether to use colors
        """
        self.color_enabled = color_enabled

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if self.color_enabled:
            return f"{color}{text}{Colors.RESET}"
        return text

    def print_header(self, text: str):
        """Print header text."""
        print()
        print(self._color("=" * 60, Colors.CYAN))
        print(self._color(text.center(60), Colors.BOLD + Colors.CYAN))
        print(self._color("=" * 60, Colors.CYAN))
        print()

    def print_section(self, text: str):
        """Print section divider."""
        print()
        print(self._color(f"--- {text} " + "-" * (56 - len(text)), Colors.BLUE))
        print()

    def print_success(self, text: str):
        """Print success message."""
        print(self._color(f"✓ {text}", Colors.GREEN))

    def print_error(self, text: str):
        """Print error message."""
        print(self._color(f"✗ {text}", Colors.RED))

    def print_warning(self, text: str):
        """Print warning message."""
        print(self._color(f"⚠ {text}", Colors.YELLOW))

    def print_info(self, text: str):
        """Print info message."""
        print(self._color(f"ℹ {text}", Colors.BLUE))

    def print_progress(self, text: str, percent: int):
        """Print progress indicator."""
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r{text}: {self._color(bar, Colors.GREEN)} {percent}%", end='', flush=True)
        if percent >= 100:
            print()  # New line when complete

    def show_main_menu(self) -> str:
        """
        Display main menu and get user choice.

        Returns:
            User's menu choice
        """
        self.print_header("AI Diagnostic Boot Drive v1.0")

        options = [
            "1. Run Full System Diagnostics",
            "2. Quick Hardware Scan",
            "3. Fix Detected Issues (Auto)",
            "4. View Diagnostic Report",
            "5. Manual Repair Mode",
            "6. View Logs",
            "7. Exit to Shell"
        ]

        for option in options:
            print(f"  {option}")

        print()
        print(self._color("=" * 60, Colors.CYAN))

        choice = input(self._color("Select option: ", Colors.BOLD)).strip()
        return choice

    def show_diagnostic_results(self, diagnostic_data: Dict):
        """Display diagnostic results."""
        self.print_header("Diagnostic Results")

        # Summary
        summary = diagnostic_data.get('summary', {})
        severity = diagnostic_data.get('severity_summary', {})

        self.print_section("Summary")
        print(f"  Total Issues Found: {self._color(str(summary.get('total_issues', 0)), Colors.YELLOW)}")
        print(f"  Critical: {self._color(str(severity.get('critical', 0)), Colors.RED)}")
        print(f"  High:     {self._color(str(severity.get('high', 0)), Colors.RED)}")
        print(f"  Medium:   {self._color(str(severity.get('medium', 0)), Colors.YELLOW)}")
        print(f"  Low:      {self._color(str(severity.get('low', 0)), Colors.GREEN)}")

        # Hardware status
        hw = diagnostic_data.get('hardware', {})
        self.print_section("Hardware Status")

        if 'cpu' in hw:
            cpu = hw['cpu']
            print(f"  CPU: {cpu.get('model', 'Unknown')[:50]}")
            if cpu.get('temperature'):
                temps = cpu['temperature']
                max_temp = max(temps)
                temp_color = Colors.RED if max_temp > 80 else Colors.YELLOW if max_temp > 70 else Colors.GREEN
                print(f"  Temperature: {self._color(f'{max_temp:.1f}°C', temp_color)}")

        if 'memory' in hw:
            mem = hw['memory']
            usage = mem.get('usage_percent', 0)
            usage_color = Colors.RED if usage > 90 else Colors.YELLOW if usage > 75 else Colors.GREEN
            print(f"  Memory: {mem.get('used_gb', 0):.1f}GB / {mem.get('total_gb', 0):.1f}GB "
                  f"({self._color(f'{usage:.1f}%', usage_color)})")

        if 'storage' in hw:
            storage = hw['storage']
            print(f"  Storage Devices: {len(storage.get('devices', []))}")

        # Issues list
        issues = diagnostic_data.get('issues', [])
        if issues:
            self.print_section("Detected Issues")
            for idx, issue in enumerate(issues[:10], 1):  # Show first 10
                severity = issue.get('severity', 'unknown').upper()
                severity_color = self._get_severity_color(severity)
                print(f"  {idx}. {self._color(f'[{severity}]', severity_color)} "
                      f"{issue.get('component', 'Unknown')}: {issue.get('issue', 'No description')}")

            if len(issues) > 10:
                print(f"\n  ... and {len(issues) - 10} more issues")

    def show_fix_summary(self, fixes: List[Dict]):
        """Display fix summary."""
        self.print_header("Available Fixes")

        for idx, fix in enumerate(fixes, 1):
            severity = fix.get('severity', 'unknown').upper()
            risk = fix.get('risk_level', 'unknown').upper()

            severity_color = self._get_severity_color(severity)
            risk_color = self._get_risk_color(risk)

            print()
            print(f"Fix #{idx}")
            print(f"  Issue: {fix.get('issue', 'Unknown')}")
            print(f"  Severity: {self._color(severity, severity_color)}")
            print(f"  Risk Level: {self._color(risk, risk_color)}")
            print(f"  Estimated Time: {fix.get('estimated_time', 'Unknown')}")
            if fix.get('explanation'):
                print(f"  Explanation: {fix['explanation'][:80]}...")

    def confirm_action(self, message: str, default: bool = False) -> bool:
        """
        Get user confirmation.

        Args:
            message: Confirmation message
            default: Default choice

        Returns:
            User's choice
        """
        default_text = "[Y/n]" if default else "[y/N]"
        prompt = f"{message} {default_text}: "

        response = input(self._color(prompt, Colors.BOLD)).strip().lower()

        if not response:
            return default

        return response in ['y', 'yes']

    def show_execution_progress(self, fix_number: int, total_fixes: int, description: str):
        """Show execution progress."""
        print()
        print(self._color(f"Executing Fix {fix_number}/{total_fixes}: {description}", Colors.BOLD))

    def show_execution_result(self, success: bool, message: str):
        """Show execution result."""
        if success:
            self.print_success(message)
        else:
            self.print_error(message)

    def show_script_preview(self, script: str, max_lines: int = 20):
        """Show script preview."""
        self.print_section("Script Preview")

        lines = script.split('\n')
        for line in lines[:max_lines]:
            print(f"  {self._color(line, Colors.GRAY)}")

        if len(lines) > max_lines:
            print(f"\n  ... ({len(lines) - max_lines} more lines)")

    def wait_for_key(self, message: str = "Press Enter to continue..."):
        """Wait for user to press a key."""
        input(self._color(f"\n{message}", Colors.CYAN))

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        severity_colors = {
            'CRITICAL': Colors.RED + Colors.BOLD,
            'HIGH': Colors.RED,
            'MEDIUM': Colors.YELLOW,
            'LOW': Colors.GREEN
        }
        return severity_colors.get(severity.upper(), Colors.WHITE)

    def _get_risk_color(self, risk: str) -> str:
        """Get color for risk level."""
        risk_colors = {
            'HIGH': Colors.RED,
            'MEDIUM': Colors.YELLOW,
            'LOW': Colors.GREEN
        }
        return risk_colors.get(risk.upper(), Colors.WHITE)

    def show_spinner(self, message: str, duration: int = 3):
        """Show animated spinner."""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        start_time = time.time()
        idx = 0

        while time.time() - start_time < duration:
            print(f"\r{self._color(spinner[idx % len(spinner)], Colors.CYAN)} {message}", end='', flush=True)
            idx += 1
            time.sleep(0.1)

        print(f"\r{' ' * (len(message) + 10)}\r", end='', flush=True)

    def show_table(self, headers: List[str], rows: List[List[str]]):
        """Display data in table format."""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(self._color(header_row, Colors.BOLD))
        print(self._color("-" * len(header_row), Colors.GRAY))

        # Print rows
        for row in rows:
            row_text = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_text)

    def clear_screen(self):
        """Clear terminal screen."""
        print("\033[2J\033[H", end='')
