#!/usr/bin/env python3
"""
AI Diagnostic Boot Drive - Main Orchestrator
Entry point for the diagnostic system.
"""

import sys
import os
import time
import json
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from diagnostics.hardware import HardwareDiagnostics
from diagnostics.software import SoftwareDiagnostics
from diagnostics.parsers import DiagnosticParser
from ai.llm_interface import LLMInterface
from ai.code_gen import CodeGenerator
from ai.validator import CodeValidator
from execution.safety import ExecutionSafety
from execution.runner import ScriptRunner
from ui.terminal import TerminalUI
from ui.reports import ReportGenerator
from utils.database import DiagnosticDatabase
from utils.logging import get_logger


class DiagnosticOrchestrator:
    """Main orchestrator for AI diagnostic system."""

    def __init__(self, config_path: str = "/config/settings.json"):
        """
        Initialize diagnostic orchestrator.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ui = TerminalUI(color_enabled=self.config['ui'].get('color_enabled', True))
        self.logger = get_logger()
        self.db = DiagnosticDatabase(self.config['storage']['database_path'])

        # Initialize components
        self.hw_diagnostics = HardwareDiagnostics(
            timeout=self.config['diagnostics'].get('timeout_per_check', 60)
        )
        self.sw_diagnostics = SoftwareDiagnostics(
            timeout=self.config['diagnostics'].get('timeout_per_check', 60)
        )
        self.parser = DiagnosticParser()
        self.llm = LLMInterface(
            model_path=self.config['llm']['model_path'],
            context_window=self.config['llm']['context_window'],
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens'],
            n_threads=self.config['llm']['n_threads']
        )
        self.code_gen = CodeGenerator()
        self.validator = CodeValidator(
            whitelist_path=self.config['security']['whitelist_path']
        )
        self.safety = ExecutionSafety(
            backup_dir=self.config['security']['backup_path']
        )
        self.runner = ScriptRunner(
            timeout=self.config['system']['max_execution_time']
        )
        self.report_gen = ReportGenerator()

        self.current_diagnostic = None
        self.current_fixes = []

    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default config
            return {
                'system': {'max_execution_time': 300},
                'llm': {
                    'model_path': '/system/model.gguf',
                    'context_window': 8192,
                    'temperature': 0.3,
                    'max_tokens': 2048,
                    'n_threads': 4
                },
                'diagnostics': {'timeout_per_check': 60},
                'storage': {'database_path': '/data/diagnostics.db'},
                'security': {
                    'whitelist_path': '/config/whitelist.json',
                    'backup_path': '/data/backups'
                },
                'ui': {'color_enabled': True},
                'execution': {
                    'dry_run_first': True,
                    'require_user_approval': True
                }
            }

    def run(self):
        """Main execution loop."""
        self.logger.main_logger.info("AI Diagnostic Boot Drive started")

        while True:
            try:
                choice = self.ui.show_main_menu()

                if choice == '1':
                    self.run_full_diagnostics()
                elif choice == '2':
                    self.run_quick_scan()
                elif choice == '3':
                    self.auto_fix_issues()
                elif choice == '4':
                    self.view_diagnostic_report()
                elif choice == '5':
                    self.manual_repair_mode()
                elif choice == '6':
                    self.view_logs()
                elif choice == '7':
                    self.ui.print_info("Exiting to shell...")
                    break
                else:
                    self.ui.print_error("Invalid option. Please try again.")

            except KeyboardInterrupt:
                self.ui.print_warning("\nOperation cancelled by user")
                if self.ui.confirm_action("Exit program?", default=False):
                    break
            except Exception as e:
                self.logger.log_error('main', f"Unexpected error: {e}", e)
                self.ui.print_error(f"Error: {e}")

        self.cleanup()

    def run_full_diagnostics(self):
        """Run complete diagnostic scan."""
        self.ui.print_header("Full System Diagnostics")
        self.logger.log_scan_start("full")

        start_time = time.time()

        # Hardware scan
        self.ui.print_info("Scanning hardware...")
        hw_data = self.hw_diagnostics.run_full_scan()
        self.ui.print_success("Hardware scan complete")

        # Software scan
        self.ui.print_info("Scanning software...")
        sw_data = self.sw_diagnostics.run_full_scan()
        self.ui.print_success("Software scan complete")

        # Aggregate results
        self.current_diagnostic = self.parser.aggregate_diagnostics(hw_data, sw_data)

        scan_duration = time.time() - start_time
        self.logger.log_scan_complete(
            "full",
            scan_duration,
            self.current_diagnostic['summary']['total_issues']
        )

        # Save to database
        diagnostic_id = self.db.insert_diagnostic(
            scan_type='full',
            hardware_info=hw_data,
            software_info=sw_data,
            issues_found=self.current_diagnostic['issues'],
            severity_summary=self.current_diagnostic['severity_summary'],
            scan_duration=scan_duration
        )

        # Display results
        self.ui.show_diagnostic_results(self.current_diagnostic)

        # AI Analysis
        if self.current_diagnostic['summary']['total_issues'] > 0:
            if self.ui.confirm_action("\nRun AI analysis for fix recommendations?", default=True):
                self.run_ai_analysis(diagnostic_id)

        self.ui.wait_for_key()

    def run_quick_scan(self):
        """Run quick hardware scan."""
        self.ui.print_header("Quick Hardware Scan")
        self.logger.log_scan_start("quick")

        start_time = time.time()

        # Quick hardware checks
        self.ui.print_info("Scanning CPU...")
        cpu_data = self.hw_diagnostics.scan_cpu()

        self.ui.print_info("Scanning memory...")
        mem_data = self.hw_diagnostics.scan_memory()

        self.ui.print_info("Scanning storage...")
        storage_data = self.hw_diagnostics.scan_storage()

        hw_data = {
            'cpu': cpu_data,
            'memory': mem_data,
            'storage': storage_data,
            'issues': [asdict(issue) for issue in self.hw_diagnostics.issues]
        }

        scan_duration = time.time() - start_time

        # Display summary
        self.ui.print_section("Quick Scan Results")
        self.ui.print_info(f"CPU: {cpu_data.get('model', 'Unknown')[:50]}")
        self.ui.print_info(f"Memory: {mem_data.get('total_gb', 0):.1f}GB ({mem_data.get('usage_percent', 0):.1f}% used)")
        self.ui.print_info(f"Storage Devices: {len(storage_data.get('devices', []))}")
        self.ui.print_info(f"Issues Found: {len(hw_data['issues'])}")

        if hw_data['issues']:
            for issue in hw_data['issues']:
                severity = issue['severity'].upper()
                self.ui.print_warning(f"[{severity}] {issue['issue']}")

        self.ui.print_success(f"\nQuick scan completed in {scan_duration:.1f}s")
        self.ui.wait_for_key()

    def run_ai_analysis(self, diagnostic_id: int):
        """Run AI analysis on diagnostic data."""
        self.ui.print_section("AI Analysis")
        self.ui.print_info("Analyzing diagnostic data with AI...")

        # Format data for LLM
        formatted_data = self.parser.format_for_llm(self.current_diagnostic)

        self.logger.log_ai_request(formatted_data[:200], "Llama 3.2 3B")

        # Call LLM
        analysis, inference_time = self.llm.analyze_diagnostics(formatted_data)

        self.logger.log_ai_response(str(analysis)[:200], inference_time)

        self.ui.print_success(f"AI analysis completed in {inference_time:.1f}s")

        # Save AI analysis
        self.db.insert_ai_analysis(
            diagnostic_id=diagnostic_id,
            prompt=formatted_data[:1000],
            response=json.dumps(analysis),
            model_name="Llama 3.2 3B",
            tokens_used=0,  # Would need to track this
            inference_time=inference_time
        )

        # Process fixes
        if 'issues' in analysis:
            self.current_fixes = self.code_gen.process_llm_analysis(analysis)

            if self.current_fixes:
                self.ui.print_success(f"Generated {len(self.current_fixes)} fix scripts")
                self.ui.show_fix_summary([asdict(fix) for fix in self.current_fixes])

                if self.ui.confirm_action("\nProceed with fixes?", default=False):
                    self.execute_fixes(diagnostic_id)
        else:
            self.ui.print_warning("No fixes generated by AI")

    def auto_fix_issues(self):
        """Automatically fix detected issues."""
        if not self.current_diagnostic:
            self.ui.print_warning("No diagnostic data available. Run diagnostics first.")
            self.ui.wait_for_key()
            return

        if not self.current_fixes:
            self.ui.print_warning("No fixes available. Run AI analysis first.")
            self.ui.wait_for_key()
            return

        self.ui.print_header("Auto Fix Issues")

        # Show summary
        self.ui.show_fix_summary([asdict(fix) for fix in self.current_fixes])

        if not self.ui.confirm_action("\nExecute all fixes?", default=False):
            return

        diagnostic_id = self.current_diagnostic.get('diagnostic_id', 0)
        self.execute_fixes(diagnostic_id)

    def execute_fixes(self, diagnostic_id: int):
        """Execute generated fix scripts."""
        self.ui.print_section("Executing Fixes")

        execution_results = []
        fixes_to_execute = self.code_gen.get_execution_order()

        for idx, fix in enumerate(fixes_to_execute, 1):
            self.ui.show_execution_progress(idx, len(fixes_to_execute), fix.issue)

            # Validate fix
            is_valid, issues = self.validator.validate_script(fix.script, fix.script_type)

            if not is_valid:
                self.ui.print_error(f"Validation failed: {', '.join(issues)}")
                self.logger.log_fix_validation(fix.issue, False, ', '.join(issues))
                continue

            self.logger.log_fix_validation(fix.issue, True)

            # Safety check
            is_safe, warnings = self.safety.verify_execution_safety(fix.script, fix.risk_level)

            if warnings:
                for warning in warnings:
                    self.ui.print_warning(warning)

            # Show script preview
            if self.ui.confirm_action("Preview script before execution?", default=True):
                self.ui.show_script_preview(fix.script)

            # Get user approval
            if self.config['execution']['require_user_approval']:
                if not self.ui.confirm_action(f"Execute this fix?", default=False):
                    self.ui.print_warning("Skipped by user")
                    continue

            # Create backup if needed
            if self.config['system'].get('backup_before_fix', True):
                self.ui.print_info("Creating backup...")
                # Implementation would backup affected files

            # Save and execute script
            script_path = self.code_gen.save_fix_to_file(fix)
            self.logger.log_execution_start(script_path, True)

            exec_start = time.time()
            result = self.runner.execute_script(
                script_path,
                dry_run=self.config['execution'].get('dry_run_first', False)
            )
            exec_duration = time.time() - exec_start

            # Log result
            self.logger.log_execution_complete(
                script_path,
                result.exit_code,
                exec_duration,
                result.success
            )

            # Save to database
            fix_id = self.db.insert_fix(
                diagnostic_id=diagnostic_id,
                issue_description=fix.issue,
                severity=fix.severity,
                script_executed=fix.script,
                script_type=fix.script_type,
                result=result.stdout[:500],
                success=result.success,
                duration=exec_duration,
                error_message=result.error_message
            )

            # Display result
            if result.success:
                self.ui.show_execution_result(True, f"Fix applied successfully")
            else:
                self.ui.show_execution_result(False, f"Fix failed: {result.error_message}")

            execution_results.append({
                'issue': fix.issue,
                'success': result.success,
                'duration': exec_duration,
                'error_message': result.error_message
            })

        # Summary
        self.ui.print_section("Execution Summary")
        summary = self.runner.get_execution_summary([])  # Would pass actual results
        self.ui.print_info(f"Total fixes attempted: {len(execution_results)}")
        successful = sum(1 for r in execution_results if r['success'])
        self.ui.print_success(f"Successful: {successful}")
        self.ui.print_error(f"Failed: {len(execution_results) - successful}")

        self.ui.wait_for_key()

    def view_diagnostic_report(self):
        """View or generate diagnostic report."""
        if not self.current_diagnostic:
            self.ui.print_warning("No diagnostic data available.")
            self.ui.wait_for_key()
            return

        self.ui.print_header("Diagnostic Report")

        # Show summary
        summary_text = self.report_gen.generate_summary_report(self.current_diagnostic)
        print(summary_text)

        # Offer to export
        print("\nExport options:")
        print("1. Text report")
        print("2. JSON report")
        print("3. HTML report")
        print("4. Back to menu")

        choice = input("\nSelect format: ").strip()

        if choice == '1':
            path = self.report_gen.generate_text_report(self.current_diagnostic)
            self.ui.print_success(f"Report saved to: {path}")
        elif choice == '2':
            path = self.report_gen.generate_json_report(self.current_diagnostic)
            self.ui.print_success(f"Report saved to: {path}")
        elif choice == '3':
            path = self.report_gen.generate_html_report(self.current_diagnostic)
            self.ui.print_success(f"Report saved to: {path}")

        self.ui.wait_for_key()

    def manual_repair_mode(self):
        """Enter manual repair mode."""
        self.ui.print_header("Manual Repair Mode")
        self.ui.print_warning("Manual repair mode not yet implemented")
        self.ui.print_info("This feature will allow manual script execution")
        self.ui.wait_for_key()

    def view_logs(self):
        """View system logs."""
        self.ui.print_header("System Logs")

        stats = self.db.get_statistics()
        self.ui.print_section("Database Statistics")
        print(f"  Total scans: {stats['total_scans']}")
        print(f"  Total fixes: {stats['total_fixes']}")
        print(f"  Successful fixes: {stats['successful_fixes']}")
        print(f"  Failed fixes: {stats['failed_fixes']}")
        print(f"  Success rate: {(stats['successful_fixes'] / stats['total_fixes'] * 100) if stats['total_fixes'] > 0 else 0:.1f}%")

        self.ui.wait_for_key()

    def cleanup(self):
        """Cleanup resources."""
        self.logger.main_logger.info("Shutting down AI Diagnostic Boot Drive")
        self.db.close()


def main():
    """Entry point."""
    # Check if running as root (required for many diagnostics)
    if os.geteuid() != 0:
        print("Warning: Not running as root. Some diagnostics may fail.")
        print("Consider running with sudo for full functionality.")
        print()

    try:
        orchestrator = DiagnosticOrchestrator()
        orchestrator.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
