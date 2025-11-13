"""
Code validator for AI-generated fix scripts.
Performs safety checks and syntax validation.
"""

import re
import json
import subprocess
from typing import Tuple, List, Dict
from pathlib import Path


class CodeValidator:
    """Validates AI-generated code for safety and correctness."""

    def __init__(self, whitelist_path: str = "/config/whitelist.json"):
        """
        Initialize code validator.

        Args:
            whitelist_path: Path to command whitelist configuration
        """
        self.whitelist_path = whitelist_path
        self.whitelist = self._load_whitelist()

    def _load_whitelist(self) -> Dict:
        """Load whitelist configuration."""
        try:
            with open(self.whitelist_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default whitelist if file not found
            return {
                'allowed_commands': [
                    'fsck', 'e2fsck', 'mount', 'umount', 'systemctl',
                    'smartctl', 'chmod', 'chown'
                ],
                'forbidden_commands': [
                    'rm -rf /', 'dd', 'mkfs', 'format'
                ],
                'forbidden_patterns': [
                    r'rm.*-rf.*/$',
                    r'dd.*of=/dev/[sh]d[a-z]$',
                    r'mkfs\\..*'
                ]
            }

    def validate_script(self, script: str, script_type: str = 'bash') -> Tuple[bool, List[str]]:
        """
        Validate a script for safety and correctness.

        Args:
            script: Script content
            script_type: Type of script (bash, python)

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for forbidden commands
        forbidden_issues = self._check_forbidden_commands(script)
        issues.extend(forbidden_issues)

        # Check for forbidden patterns
        pattern_issues = self._check_forbidden_patterns(script)
        issues.extend(pattern_issues)

        # Check for dangerous operations
        danger_issues = self._check_dangerous_operations(script)
        issues.extend(danger_issues)

        # Syntax check
        syntax_issues = self._check_syntax(script, script_type)
        issues.extend(syntax_issues)

        # Check for required safety measures
        safety_issues = self._check_safety_measures(script, script_type)
        issues.extend(safety_issues)

        is_valid = len(issues) == 0

        return is_valid, issues

    def _check_forbidden_commands(self, script: str) -> List[str]:
        """Check for explicitly forbidden commands."""
        issues = []
        forbidden = self.whitelist.get('forbidden_commands', [])

        for forbidden_cmd in forbidden:
            if forbidden_cmd in script:
                issues.append(f"Forbidden command detected: {forbidden_cmd}")

        return issues

    def _check_forbidden_patterns(self, script: str) -> List[str]:
        """Check for forbidden regex patterns."""
        issues = []
        patterns = self.whitelist.get('forbidden_patterns', [])

        for pattern in patterns:
            if re.search(pattern, script, re.MULTILINE):
                issues.append(f"Forbidden pattern detected: {pattern}")

        return issues

    def _check_dangerous_operations(self, script: str) -> List[str]:
        """Check for potentially dangerous operations."""
        issues = []

        dangerous_checks = [
            (r'\brm\s+.*-rf', 'Recursive force delete detected'),
            (r'\bdd\b.*\bif=', 'Direct disk write detected (dd command)'),
            (r'\bmkfs\b', 'Filesystem formatting detected'),
            (r'\bformat\b', 'Format command detected'),
            (r'>\s*/dev/sd[a-z]', 'Direct write to disk device'),
            (r'/dev/null.*>.*/', 'Potential data destruction'),
            (r'chmod\s+777', 'Overly permissive permissions (777)'),
            (r'chown\s+-R\s+.*:.*\s+/', 'Recursive ownership change on root'),
            (r'wget.*\|\s*sh', 'Piping downloaded content to shell'),
            (r'curl.*\|\s*bash', 'Piping downloaded content to bash'),
            (r'eval\s+\$\(', 'Eval with command substitution (potential injection)'),
            (r':\(\)\{\s*:\|:&\s*\};:', 'Fork bomb detected'),
        ]

        for pattern, description in dangerous_checks:
            if re.search(pattern, script, re.IGNORECASE):
                issues.append(description)

        return issues

    def _check_syntax(self, script: str, script_type: str) -> List[str]:
        """Check script syntax."""
        issues = []

        if script_type == 'bash':
            issues.extend(self._check_bash_syntax(script))
        elif script_type == 'python':
            issues.extend(self._check_python_syntax(script))

        return issues

    def _check_bash_syntax(self, script: str) -> List[str]:
        """Check bash script syntax."""
        issues = []

        try:
            # Use bash -n to check syntax without executing
            result = subprocess.run(
                ['bash', '-n'],
                input=script,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                issues.append(f"Bash syntax error: {result.stderr}")

        except subprocess.TimeoutExpired:
            issues.append("Syntax check timed out")
        except FileNotFoundError:
            # bash not available, skip syntax check
            pass
        except Exception as e:
            issues.append(f"Syntax check failed: {str(e)}")

        return issues

    def _check_python_syntax(self, script: str) -> List[str]:
        """Check Python script syntax."""
        issues = []

        try:
            compile(script, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Python syntax error: {str(e)}")
        except Exception as e:
            issues.append(f"Python compilation error: {str(e)}")

        return issues

    def _check_safety_measures(self, script: str, script_type: str) -> List[str]:
        """Check for recommended safety measures."""
        warnings = []

        # Check for error handling
        if script_type == 'bash':
            if 'set -e' not in script and 'set -o errexit' not in script:
                warnings.append("Warning: Script lacks 'set -e' for error handling")

            if '||' not in script and '&&' not in script:
                warnings.append("Warning: No error handling operators (|| or &&) found")

        elif script_type == 'python':
            if 'try:' not in script and 'except' not in script:
                warnings.append("Warning: No try-except error handling found")

        return warnings

    def check_command_whitelist(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is in the whitelist.

        Args:
            command: Command to check

        Returns:
            Tuple of (is_allowed, reason)
        """
        allowed = self.whitelist.get('allowed_commands', [])

        # Extract base command (first word)
        base_command = command.split()[0] if command.strip() else ''

        if base_command in allowed:
            return True, "Command is whitelisted"

        # Check if it's a common safe command
        safe_commands = ['echo', 'ls', 'cat', 'grep', 'awk', 'sed', 'test', '[']
        if base_command in safe_commands:
            return True, "Command is considered safe"

        return False, f"Command '{base_command}' is not in whitelist"

    def validate_file_operations(self, script: str) -> Tuple[bool, List[str]]:
        """
        Validate file operations in script.

        Args:
            script: Script content

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []
        restricted_paths = self.whitelist.get('restricted_paths', [])

        # Check for operations on restricted paths
        for path in restricted_paths:
            if path in script:
                issues.append(f"Operation on restricted path detected: {path}")

        # Check for writes to critical directories
        critical_dirs = ['/boot', '/etc', '/sys', '/proc']
        for dir_path in critical_dirs:
            # Look for redirects or writes to these directories
            if re.search(f'>.*{dir_path}', script):
                issues.append(f"Write to critical directory detected: {dir_path}")

        is_safe = len(issues) == 0
        return is_safe, issues

    def get_risk_level(self, script: str) -> str:
        """
        Assess overall risk level of script.

        Args:
            script: Script content

        Returns:
            Risk level: low, medium, or high
        """
        is_valid, issues = self.validate_script(script)

        if not is_valid:
            # Has validation issues - high risk
            return 'high'

        # Check for potentially risky operations
        risky_patterns = [
            r'\bchmod\b',
            r'\bchown\b',
            r'\bmount\b',
            r'\bumount\b',
            r'\bfsck\b',
            r'/dev/',
            r'\bsystemctl\b.*restart',
            r'\bsystemctl\b.*stop',
        ]

        risky_count = sum(1 for pattern in risky_patterns if re.search(pattern, script))

        if risky_count >= 3:
            return 'medium'
        elif risky_count >= 1:
            return 'low'
        else:
            return 'low'

    def generate_validation_report(self, script: str, script_type: str = 'bash') -> Dict:
        """
        Generate comprehensive validation report.

        Args:
            script: Script content
            script_type: Type of script

        Returns:
            Validation report dictionary
        """
        is_valid, issues = self.validate_script(script, script_type)
        is_safe_files, file_issues = self.validate_file_operations(script)
        risk_level = self.get_risk_level(script)

        report = {
            'valid': is_valid,
            'safe': is_valid and is_safe_files,
            'risk_level': risk_level,
            'issues': issues,
            'file_operation_issues': file_issues,
            'total_issues': len(issues) + len(file_issues),
            'recommendations': []
        }

        # Add recommendations
        if not is_valid:
            report['recommendations'].append("Do not execute this script - validation failed")
        elif risk_level == 'high':
            report['recommendations'].append("High risk - require manual review before execution")
        elif risk_level == 'medium':
            report['recommendations'].append("Medium risk - backup data before execution")
        else:
            report['recommendations'].append("Low risk - safe to execute with monitoring")

        return report
