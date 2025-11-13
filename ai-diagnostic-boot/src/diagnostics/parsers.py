"""
Diagnostic parsers and data aggregation.
Consolidates hardware and software diagnostic data for AI analysis.
"""

import json
from typing import Dict, List
from datetime import datetime


class DiagnosticParser:
    """Parses and aggregates diagnostic data."""

    @staticmethod
    def aggregate_diagnostics(hardware_data: Dict, software_data: Dict) -> Dict:
        """
        Aggregate hardware and software diagnostic data.

        Args:
            hardware_data: Hardware diagnostic results
            software_data: Software diagnostic results

        Returns:
            Aggregated diagnostic report
        """
        # Combine all issues
        all_issues = []
        all_issues.extend(hardware_data.get('issues', []))
        all_issues.extend(software_data.get('issues', []))

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 4))

        # Calculate severity summary
        severity_summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'total': len(all_issues)
        }

        for issue in all_issues:
            severity = issue.get('severity', 'low').lower()
            if severity in severity_summary:
                severity_summary[severity] += 1

        # Build aggregated report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(all_issues),
                'severity_breakdown': severity_summary,
                'hardware_checks_completed': len(hardware_data.keys()) - 1,  # -1 for 'issues' key
                'software_checks_completed': len(software_data.keys()) - 1
            },
            'hardware': hardware_data,
            'software': software_data,
            'issues': all_issues,
            'severity_summary': severity_summary
        }

        return report

    @staticmethod
    def format_for_llm(diagnostic_report: Dict) -> str:
        """
        Format diagnostic report for LLM consumption.

        Args:
            diagnostic_report: Aggregated diagnostic report

        Returns:
            Formatted string optimized for LLM analysis
        """
        output = []

        # System overview
        output.append("=== SYSTEM DIAGNOSTIC REPORT ===\n")
        output.append(f"Timestamp: {diagnostic_report['timestamp']}\n")
        output.append(f"Total Issues Found: {diagnostic_report['summary']['total_issues']}\n")

        severity = diagnostic_report['severity_summary']
        output.append(f"Severity Breakdown: Critical={severity['critical']}, "
                     f"High={severity['high']}, Medium={severity['medium']}, "
                     f"Low={severity['low']}\n\n")

        # Hardware summary
        hw = diagnostic_report['hardware']
        output.append("=== HARDWARE STATUS ===\n")

        if 'cpu' in hw:
            cpu = hw['cpu']
            output.append(f"CPU: {cpu.get('model', 'Unknown')}\n")
            output.append(f"  Cores: {cpu.get('cores', 0)}, Threads: {cpu.get('threads', 0)}\n")
            if cpu.get('temperature'):
                temps = cpu['temperature']
                output.append(f"  Temperature: {max(temps):.1f}°C (max)\n")

        if 'memory' in hw:
            mem = hw['memory']
            output.append(f"Memory: {mem.get('used_gb', 0):.1f}GB / {mem.get('total_gb', 0):.1f}GB "
                         f"({mem.get('usage_percent', 0):.1f}% used)\n")
            if mem.get('swap_total_gb', 0) > 0:
                output.append(f"  Swap: {mem.get('swap_used_gb', 0):.1f}GB / "
                             f"{mem.get('swap_total_gb', 0):.1f}GB\n")

        if 'storage' in hw:
            storage = hw['storage']
            output.append(f"Storage Devices: {len(storage.get('devices', []))}\n")
            for device in storage.get('devices', [])[:5]:  # Limit to first 5
                name = device.get('name', 'unknown')
                size = device.get('size', 'unknown')
                output.append(f"  - {name}: {size}\n")
                smart = device.get('smart', {})
                if smart.get('available'):
                    health = "HEALTHY" if smart.get('healthy') else "FAILING"
                    output.append(f"    SMART: {health}\n")

        if 'network' in hw:
            network = hw['network']
            output.append(f"Network Interfaces: {len(network.get('interfaces', []))}\n")
            for iface in network.get('interfaces', []):
                output.append(f"  - {iface['name']}: {iface['state']}\n")

        output.append("\n")

        # Software summary
        sw = diagnostic_report['software']
        output.append("=== SOFTWARE STATUS ===\n")

        if 'bootloader' in sw:
            bl = sw['bootloader']
            output.append(f"Bootloader: {bl.get('type', 'unknown')} ({bl.get('boot_mode', 'unknown')})\n")

        if 'filesystems' in sw:
            fs = sw['filesystems']
            output.append(f"Mounted Filesystems: {len(fs.get('mounted', []))}\n")

        if 'logs' in sw:
            logs = sw['logs']
            output.append(f"Recent Errors in Logs: {logs.get('errors_found', 0)}\n")
            output.append(f"Critical Issues in Logs: {logs.get('critical_found', 0)}\n")

        if 'processes' in sw:
            procs = sw['processes']
            output.append(f"Running Processes: {procs.get('total', 0)}\n")
            if procs.get('zombie', 0) > 0:
                output.append(f"  Zombie Processes: {procs['zombie']}\n")

        output.append("\n")

        # Detailed issues
        output.append("=== DETECTED ISSUES (by severity) ===\n\n")

        for issue in diagnostic_report['issues']:
            severity = issue.get('severity', 'unknown').upper()
            component = issue.get('component', 'Unknown')
            description = issue.get('issue', 'No description')
            recommendation = issue.get('recommendation', 'No recommendation')

            output.append(f"[{severity}] {component}: {description}\n")
            output.append(f"  Recommendation: {recommendation}\n")

            # Include log excerpts if available
            log_excerpt = issue.get('log_excerpt', '')
            if log_excerpt:
                output.append(f"  Log excerpt: {log_excerpt[:150]}...\n")

            output.append("\n")

        return ''.join(output)

    @staticmethod
    def format_issue_list(issues: List[Dict]) -> str:
        """
        Format issue list in a concise format.

        Args:
            issues: List of issue dictionaries

        Returns:
            Formatted string
        """
        if not issues:
            return "No issues detected."

        output = []
        for idx, issue in enumerate(issues, 1):
            severity = issue.get('severity', 'unknown').upper()
            component = issue.get('component', 'Unknown')
            description = issue.get('issue', 'No description')
            output.append(f"{idx}. [{severity}] {component}: {description}")

        return '\n'.join(output)

    @staticmethod
    def extract_critical_issues(diagnostic_report: Dict) -> List[Dict]:
        """
        Extract only critical and high severity issues.

        Args:
            diagnostic_report: Aggregated diagnostic report

        Returns:
            List of critical/high issues
        """
        all_issues = diagnostic_report.get('issues', [])
        critical_issues = [
            issue for issue in all_issues
            if issue.get('severity', '').lower() in ['critical', 'high']
        ]
        return critical_issues

    @staticmethod
    def generate_summary_stats(diagnostic_report: Dict) -> Dict:
        """
        Generate summary statistics from diagnostic report.

        Args:
            diagnostic_report: Aggregated diagnostic report

        Returns:
            Summary statistics dictionary
        """
        stats = {
            'total_issues': diagnostic_report['summary']['total_issues'],
            'critical_count': diagnostic_report['severity_summary']['critical'],
            'high_count': diagnostic_report['severity_summary']['high'],
            'medium_count': diagnostic_report['severity_summary']['medium'],
            'low_count': diagnostic_report['severity_summary']['low'],
            'requires_immediate_attention': False,
            'health_score': 100,  # Start at 100
            'components_checked': (
                diagnostic_report['summary']['hardware_checks_completed'] +
                diagnostic_report['summary']['software_checks_completed']
            )
        }

        # Calculate health score (simple scoring)
        stats['health_score'] -= stats['critical_count'] * 20
        stats['health_score'] -= stats['high_count'] * 10
        stats['health_score'] -= stats['medium_count'] * 5
        stats['health_score'] -= stats['low_count'] * 2
        stats['health_score'] = max(0, stats['health_score'])

        # Flag if immediate attention required
        if stats['critical_count'] > 0 or stats['high_count'] > 2:
            stats['requires_immediate_attention'] = True

        return stats

    @staticmethod
    def to_json(diagnostic_report: Dict, pretty: bool = True) -> str:
        """
        Convert diagnostic report to JSON string.

        Args:
            diagnostic_report: Aggregated diagnostic report
            pretty: Whether to pretty-print JSON

        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(diagnostic_report, indent=2, default=str)
        return json.dumps(diagnostic_report, default=str)

    @staticmethod
    def from_json(json_string: str) -> Dict:
        """
        Parse diagnostic report from JSON string.

        Args:
            json_string: JSON string

        Returns:
            Diagnostic report dictionary
        """
        return json.loads(json_string)
