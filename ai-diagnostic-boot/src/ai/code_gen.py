"""
AI-powered code generation for fixes.
Processes LLM output and generates executable fix scripts.
"""

import re
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile


@dataclass
class GeneratedFix:
    """Represents a generated fix script."""
    issue_id: str
    issue: str
    severity: str
    script: str
    script_type: str  # bash, python
    risk_level: str
    estimated_time: str
    explanation: str
    validated: bool = False
    validation_notes: str = ""


class CodeGenerator:
    """Generates and manages fix scripts from AI analysis."""

    def __init__(self, output_dir: str = "/tmp/ai-fixes"):
        """
        Initialize code generator.

        Args:
            output_dir: Directory to store generated scripts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_fixes: List[GeneratedFix] = []

    def process_llm_analysis(self, llm_output: Dict) -> List[GeneratedFix]:
        """
        Process LLM analysis and extract fix scripts.

        Args:
            llm_output: LLM analysis output

        Returns:
            List of GeneratedFix objects
        """
        fixes = []

        issues = llm_output.get('issues', [])

        for idx, issue_data in enumerate(issues):
            fix = self._create_fix_from_issue(issue_data, idx)
            if fix:
                fixes.append(fix)
                self.generated_fixes.append(fix)

        return fixes

    def _create_fix_from_issue(self, issue_data: Dict, index: int) -> Optional[GeneratedFix]:
        """
        Create a GeneratedFix from issue data.

        Args:
            issue_data: Issue dictionary from LLM
            index: Issue index

        Returns:
            GeneratedFix object or None if invalid
        """
        issue_desc = issue_data.get('issue', 'Unknown issue')
        script = issue_data.get('fix_script', '')

        if not script or not script.strip():
            # No fix script provided
            return None

        # Generate unique ID
        issue_id = hashlib.md5(f"{issue_desc}{script}".encode()).hexdigest()[:12]

        # Detect script type
        script_type = self._detect_script_type(script)

        fix = GeneratedFix(
            issue_id=issue_id,
            issue=issue_desc,
            severity=issue_data.get('severity', 'medium'),
            script=script,
            script_type=script_type,
            risk_level=issue_data.get('risk_level', 'medium'),
            estimated_time=issue_data.get('estimated_time', 'Unknown'),
            explanation=issue_data.get('diagnosis', '')
        )

        return fix

    def _detect_script_type(self, script: str) -> str:
        """
        Detect script type from shebang or content.

        Args:
            script: Script content

        Returns:
            Script type (bash, python, etc.)
        """
        first_line = script.strip().split('\n')[0]

        if '#!/bin/bash' in first_line or '#!/bin/sh' in first_line:
            return 'bash'
        elif '#!/usr/bin/python' in first_line or '#!/usr/bin/env python' in first_line:
            return 'python'
        elif 'import ' in script or 'def ' in script:
            return 'python'
        else:
            return 'bash'  # Default to bash

    def save_fix_to_file(self, fix: GeneratedFix) -> str:
        """
        Save fix script to file.

        Args:
            fix: GeneratedFix object

        Returns:
            Path to saved file
        """
        # Determine file extension
        ext = '.sh' if fix.script_type == 'bash' else '.py'
        filename = f"fix_{fix.issue_id}{ext}"
        filepath = self.output_dir / filename

        # Add shebang if missing
        script = fix.script
        if not script.startswith('#!'):
            if fix.script_type == 'bash':
                script = '#!/bin/bash\n' + script
            elif fix.script_type == 'python':
                script = '#!/usr/bin/env python3\n' + script

        # Write to file
        with open(filepath, 'w') as f:
            f.write(script)

        # Make executable
        filepath.chmod(0o755)

        return str(filepath)

    def get_fix_by_id(self, issue_id: str) -> Optional[GeneratedFix]:
        """
        Get fix by issue ID.

        Args:
            issue_id: Issue identifier

        Returns:
            GeneratedFix or None
        """
        for fix in self.generated_fixes:
            if fix.issue_id == issue_id:
                return fix
        return None

    def get_fixes_by_severity(self, severity: str) -> List[GeneratedFix]:
        """
        Get all fixes of a specific severity.

        Args:
            severity: Severity level

        Returns:
            List of GeneratedFix objects
        """
        return [
            fix for fix in self.generated_fixes
            if fix.severity.lower() == severity.lower()
        ]

    def mark_validated(self, issue_id: str, notes: str = ""):
        """
        Mark a fix as validated.

        Args:
            issue_id: Issue identifier
            notes: Validation notes
        """
        fix = self.get_fix_by_id(issue_id)
        if fix:
            fix.validated = True
            fix.validation_notes = notes

    def generate_fix_summary(self) -> Dict:
        """
        Generate summary of all fixes.

        Returns:
            Summary dictionary
        """
        summary = {
            'total_fixes': len(self.generated_fixes),
            'by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'by_risk': {
                'low': 0,
                'medium': 0,
                'high': 0
            },
            'validated': 0,
            'not_validated': 0
        }

        for fix in self.generated_fixes:
            # Count by severity
            sev = fix.severity.lower()
            if sev in summary['by_severity']:
                summary['by_severity'][sev] += 1

            # Count by risk
            risk = fix.risk_level.lower()
            if risk in summary['by_risk']:
                summary['by_risk'][risk] += 1

            # Count validation status
            if fix.validated:
                summary['validated'] += 1
            else:
                summary['not_validated'] += 1

        return summary

    def export_fixes_to_json(self) -> List[Dict]:
        """
        Export all fixes to JSON-serializable format.

        Returns:
            List of fix dictionaries
        """
        return [asdict(fix) for fix in self.generated_fixes]

    def create_dry_run_script(self, fix: GeneratedFix) -> str:
        """
        Create a dry-run version of the script that only shows what would be done.

        Args:
            fix: GeneratedFix object

        Returns:
            Path to dry-run script
        """
        # Wrap script in echo commands to show what would execute
        dry_run_script = "#!/bin/bash\n"
        dry_run_script += "echo '=== DRY RUN MODE - No changes will be made ==='\n"
        dry_run_script += "echo ''\n"
        dry_run_script += f"echo 'Issue: {fix.issue}'\n"
        dry_run_script += f"echo 'Severity: {fix.severity}'\n"
        dry_run_script += f"echo 'Risk Level: {fix.risk_level}'\n"
        dry_run_script += "echo ''\n"
        dry_run_script += "echo 'Commands that would be executed:'\n"
        dry_run_script += "echo '-----------------------------------'\n"

        # Extract commands from original script
        for line in fix.script.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Escape quotes for echo
                escaped_line = line.replace("'", "'\\''")
                dry_run_script += f"echo '  {escaped_line}'\n"

        dry_run_script += "echo '-----------------------------------'\n"

        # Save dry-run script
        filename = f"dry_run_{fix.issue_id}.sh"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(dry_run_script)

        filepath.chmod(0o755)

        return str(filepath)

    def cleanup_generated_files(self):
        """Remove all generated fix scripts."""
        for filepath in self.output_dir.glob('fix_*.sh'):
            filepath.unlink()
        for filepath in self.output_dir.glob('fix_*.py'):
            filepath.unlink()
        for filepath in self.output_dir.glob('dry_run_*.sh'):
            filepath.unlink()

        self.generated_fixes.clear()

    def get_execution_order(self) -> List[GeneratedFix]:
        """
        Get recommended execution order for fixes (critical first).

        Returns:
            Ordered list of GeneratedFix objects
        """
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        return sorted(
            self.generated_fixes,
            key=lambda f: severity_order.get(f.severity.lower(), 4)
        )
