"""
Report generation module.
Creates diagnostic reports in various formats (text, JSON, HTML).
"""

import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path


class ReportGenerator:
    """Generates diagnostic reports."""

    def __init__(self, output_dir: str = "/data/reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(self, diagnostic_data: Dict,
                            execution_results: List[Dict] = None) -> str:
        """
        Generate plain text diagnostic report.

        Args:
            diagnostic_data: Diagnostic scan results
            execution_results: Optional fix execution results

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_path = self.output_dir / f"diagnostic_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("AI DIAGNOSTIC BOOT DRIVE - SYSTEM REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = diagnostic_data.get('summary', {})
            severity = diagnostic_data.get('severity_summary', {})

            f.write(f"Total Issues Found: {summary.get('total_issues', 0)}\n")
            f.write(f"Critical Issues: {severity.get('critical', 0)}\n")
            f.write(f"High Priority Issues: {severity.get('high', 0)}\n")
            f.write(f"Medium Priority Issues: {severity.get('medium', 0)}\n")
            f.write(f"Low Priority Issues: {severity.get('low', 0)}\n\n")

            # Hardware Status
            f.write("HARDWARE STATUS\n")
            f.write("-" * 80 + "\n")
            hw = diagnostic_data.get('hardware', {})

            if 'cpu' in hw:
                cpu = hw['cpu']
                f.write(f"CPU Model: {cpu.get('model', 'Unknown')}\n")
                f.write(f"CPU Cores: {cpu.get('cores', 0)}\n")
                f.write(f"CPU Threads: {cpu.get('threads', 0)}\n")
                if cpu.get('temperature'):
                    temps = cpu['temperature']
                    f.write(f"CPU Temperature: {max(temps):.1f}°C (max)\n")

            if 'memory' in hw:
                mem = hw['memory']
                f.write(f"\nMemory Total: {mem.get('total_gb', 0):.2f} GB\n")
                f.write(f"Memory Used: {mem.get('used_gb', 0):.2f} GB\n")
                f.write(f"Memory Usage: {mem.get('usage_percent', 0):.1f}%\n")

            if 'storage' in hw:
                storage = hw['storage']
                f.write(f"\nStorage Devices: {len(storage.get('devices', []))}\n")
                for device in storage.get('devices', []):
                    f.write(f"  - {device.get('name')}: {device.get('size', 'Unknown')}\n")
                    smart = device.get('smart', {})
                    if smart.get('available'):
                        status = "HEALTHY" if smart.get('healthy') else "FAILING"
                        f.write(f"    SMART Status: {status}\n")

            f.write("\n")

            # Software Status
            f.write("SOFTWARE STATUS\n")
            f.write("-" * 80 + "\n")
            sw = diagnostic_data.get('software', {})

            if 'bootloader' in sw:
                bl = sw['bootloader']
                f.write(f"Bootloader: {bl.get('type', 'Unknown')}\n")
                f.write(f"Boot Mode: {bl.get('boot_mode', 'Unknown')}\n")

            if 'filesystems' in sw:
                fs = sw['filesystems']
                f.write(f"Mounted Filesystems: {len(fs.get('mounted', []))}\n")

            if 'logs' in sw:
                logs = sw['logs']
                f.write(f"Recent Errors: {logs.get('errors_found', 0)}\n")
                f.write(f"Critical Issues: {logs.get('critical_found', 0)}\n")

            if 'processes' in sw:
                procs = sw['processes']
                f.write(f"Running Processes: {procs.get('total', 0)}\n")

            f.write("\n")

            # Detailed Issues
            issues = diagnostic_data.get('issues', [])
            if issues:
                f.write("DETAILED ISSUES\n")
                f.write("-" * 80 + "\n\n")

                for idx, issue in enumerate(issues, 1):
                    f.write(f"Issue #{idx}\n")
                    f.write(f"  Component: {issue.get('component', 'Unknown')}\n")
                    f.write(f"  Severity: {issue.get('severity', 'Unknown').upper()}\n")
                    f.write(f"  Description: {issue.get('issue', 'No description')}\n")
                    f.write(f"  Recommendation: {issue.get('recommendation', 'No recommendation')}\n")
                    f.write("\n")

            # Execution Results
            if execution_results:
                f.write("FIX EXECUTION RESULTS\n")
                f.write("-" * 80 + "\n\n")

                for idx, result in enumerate(execution_results, 1):
                    f.write(f"Fix #{idx}\n")
                    f.write(f"  Issue: {result.get('issue', 'Unknown')}\n")
                    f.write(f"  Success: {'YES' if result.get('success') else 'NO'}\n")
                    f.write(f"  Duration: {result.get('duration', 0):.2f}s\n")
                    if not result.get('success'):
                        f.write(f"  Error: {result.get('error_message', 'Unknown error')}\n")
                    f.write("\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        return str(report_path)

    def generate_json_report(self, diagnostic_data: Dict,
                            execution_results: List[Dict] = None) -> str:
        """
        Generate JSON diagnostic report.

        Args:
            diagnostic_data: Diagnostic scan results
            execution_results: Optional fix execution results

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_path = self.output_dir / f"diagnostic_report_{timestamp}.json"

        report = {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'diagnostic_data': diagnostic_data,
            'execution_results': execution_results or []
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return str(report_path)

    def generate_html_report(self, diagnostic_data: Dict,
                            execution_results: List[Dict] = None) -> str:
        """
        Generate HTML diagnostic report.

        Args:
            diagnostic_data: Diagnostic scan results
            execution_results: Optional fix execution results

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_path = self.output_dir / f"diagnostic_report_{timestamp}.html"

        html = self._build_html_report(diagnostic_data, execution_results)

        with open(report_path, 'w') as f:
            f.write(html)

        return str(report_path)

    def _build_html_report(self, diagnostic_data: Dict,
                          execution_results: List[Dict] = None) -> str:
        """Build HTML report content."""

        summary = diagnostic_data.get('summary', {})
        severity = diagnostic_data.get('severity_summary', {})
        issues = diagnostic_data.get('issues', [])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Diagnostic Boot Drive - System Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #7f8c8d;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .critical {{ background: #e74c3c; color: white; }}
        .high {{ background: #e67e22; color: white; }}
        .medium {{ background: #f39c12; color: white; }}
        .low {{ background: #27ae60; color: white; }}
        .info {{ background: #3498db; color: white; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #34495e;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Diagnostic Boot Drive - System Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-card info">
                <h3>Total Issues</h3>
                <div class="value">{summary.get('total_issues', 0)}</div>
            </div>
            <div class="summary-card critical">
                <h3>Critical</h3>
                <div class="value">{severity.get('critical', 0)}</div>
            </div>
            <div class="summary-card high">
                <h3>High</h3>
                <div class="value">{severity.get('high', 0)}</div>
            </div>
            <div class="summary-card medium">
                <h3>Medium</h3>
                <div class="value">{severity.get('medium', 0)}</div>
            </div>
            <div class="summary-card low">
                <h3>Low</h3>
                <div class="value">{severity.get('low', 0)}</div>
            </div>
        </div>

        <h2>Detected Issues</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Component</th>
                    <th>Severity</th>
                    <th>Issue</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
"""

        for idx, issue in enumerate(issues, 1):
            severity_class = issue.get('severity', 'low').lower()
            html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{issue.get('component', 'Unknown')}</td>
                    <td><span class="badge {severity_class}">{issue.get('severity', 'Unknown').upper()}</span></td>
                    <td>{issue.get('issue', 'No description')}</td>
                    <td>{issue.get('recommendation', 'No recommendation')}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        if execution_results:
            html += """
        <h2>Fix Execution Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Issue</th>
                    <th>Status</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>
"""
            for idx, result in enumerate(execution_results, 1):
                status = "SUCCESS" if result.get('success') else "FAILED"
                status_class = "low" if result.get('success') else "critical"
                html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{result.get('issue', 'Unknown')}</td>
                    <td><span class="badge {status_class}">{status}</span></td>
                    <td>{result.get('duration', 0):.2f}s</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""

        html += """
    </div>
</body>
</html>
"""

        return html

    def generate_summary_report(self, diagnostic_data: Dict) -> str:
        """
        Generate brief summary report.

        Args:
            diagnostic_data: Diagnostic scan results

        Returns:
            Summary text
        """
        summary = diagnostic_data.get('summary', {})
        severity = diagnostic_data.get('severity_summary', {})

        lines = [
            "=== DIAGNOSTIC SUMMARY ===",
            f"Total Issues: {summary.get('total_issues', 0)}",
            f"Critical: {severity.get('critical', 0)}",
            f"High: {severity.get('high', 0)}",
            f"Medium: {severity.get('medium', 0)}",
            f"Low: {severity.get('low', 0)}",
            "========================="
        ]

        return "\n".join(lines)
