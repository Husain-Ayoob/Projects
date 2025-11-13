"""
Logging utilities for AI Diagnostic Boot Drive.
Provides structured logging with rotation and export capabilities.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class DiagnosticLogger:
    """Custom logger for diagnostic operations."""

    def __init__(self, log_dir: str = "/data/logs",
                 max_size_mb: int = 100,
                 backup_count: int = 5):
        """
        Initialize diagnostic logger.

        Args:
            log_dir: Directory for log files
            max_size_mb: Maximum size per log file in MB
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_size = max_size_mb * 1024 * 1024
        self.backup_count = backup_count

        # Create separate loggers for different components
        self.main_logger = self._create_logger('main', 'diagnostic-main.log')
        self.scan_logger = self._create_logger('scan', 'diagnostic-scan.log')
        self.fix_logger = self._create_logger('fix', 'diagnostic-fix.log')
        self.execution_logger = self._create_logger('execution', 'execution.log')
        self.ai_logger = self._create_logger('ai', 'ai-analysis.log')

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a logger with rotating file handler."""
        logger = logging.getLogger(f'diagnostic.{name}')
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_scan_start(self, scan_type: str):
        """Log the start of a diagnostic scan."""
        self.scan_logger.info(f"Starting {scan_type} scan")

    def log_scan_complete(self, scan_type: str, duration: float, issues_found: int):
        """Log scan completion."""
        self.scan_logger.info(
            f"Completed {scan_type} scan in {duration:.2f}s - "
            f"Found {issues_found} issues"
        )

    def log_hardware_check(self, component: str, status: str, details: Dict):
        """Log hardware check results."""
        self.scan_logger.debug(
            f"Hardware check - {component}: {status} - {json.dumps(details)}"
        )

    def log_software_check(self, component: str, status: str, details: Dict):
        """Log software check results."""
        self.scan_logger.debug(
            f"Software check - {component}: {status} - {json.dumps(details)}"
        )

    def log_issue_detected(self, severity: str, issue: str, details: Dict):
        """Log detected issue."""
        log_method = {
            'critical': self.scan_logger.critical,
            'high': self.scan_logger.error,
            'medium': self.scan_logger.warning,
            'low': self.scan_logger.info
        }.get(severity.lower(), self.scan_logger.info)

        log_method(f"Issue detected ({severity}): {issue} - {json.dumps(details)}")

    def log_ai_request(self, prompt: str, model: str):
        """Log AI analysis request."""
        self.ai_logger.info(f"AI request to {model}")
        self.ai_logger.debug(f"Prompt: {prompt[:200]}...")

    def log_ai_response(self, response: str, inference_time: float):
        """Log AI analysis response."""
        self.ai_logger.info(f"AI response received in {inference_time:.2f}s")
        self.ai_logger.debug(f"Response: {response[:200]}...")

    def log_fix_generated(self, issue: str, script: str):
        """Log generated fix script."""
        self.fix_logger.info(f"Fix generated for: {issue}")
        self.fix_logger.debug(f"Script:\n{script}")

    def log_fix_validation(self, issue: str, valid: bool, reason: str = ""):
        """Log fix validation result."""
        if valid:
            self.fix_logger.info(f"Fix validated for: {issue}")
        else:
            self.fix_logger.warning(f"Fix validation failed for: {issue} - {reason}")

    def log_execution_start(self, command: str, approved: bool):
        """Log script execution start."""
        approval_status = "user-approved" if approved else "auto-approved"
        self.execution_logger.info(f"Executing command ({approval_status}): {command}")

    def log_execution_complete(self, command: str, exit_code: int,
                              duration: float, success: bool):
        """Log script execution completion."""
        status = "SUCCESS" if success else "FAILED"
        self.execution_logger.info(
            f"Execution {status} - Command: {command} - "
            f"Exit code: {exit_code} - Duration: {duration:.2f}s"
        )

    def log_execution_output(self, command: str, stdout: str, stderr: str):
        """Log execution output."""
        self.execution_logger.debug(f"STDOUT for {command}:\n{stdout}")
        if stderr:
            self.execution_logger.debug(f"STDERR for {command}:\n{stderr}")

    def log_error(self, component: str, error: str, exception: Optional[Exception] = None):
        """Log error with optional exception."""
        self.main_logger.error(f"{component} error: {error}")
        if exception:
            self.main_logger.exception(exception)

    def log_security_violation(self, command: str, reason: str):
        """Log security violation."""
        self.execution_logger.critical(
            f"SECURITY VIOLATION - Command blocked: {command} - Reason: {reason}"
        )

    def log_backup_created(self, target: str, backup_path: str):
        """Log backup creation."""
        self.fix_logger.info(f"Backup created: {target} -> {backup_path}")

    def log_rollback(self, target: str, reason: str):
        """Log rollback operation."""
        self.fix_logger.warning(f"Rollback initiated for {target} - Reason: {reason}")

    def create_session_log(self, session_id: str) -> str:
        """
        Create a timestamped session log file.

        Args:
            session_id: Unique session identifier

        Returns:
            Path to session log file
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = self.log_dir / f"session-{timestamp}-{session_id}.log"

        # Create session-specific logger
        session_logger = logging.getLogger(f'diagnostic.session.{session_id}')
        session_logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        session_logger.addHandler(handler)

        session_logger.info(f"Session {session_id} started")

        return str(log_file)

    def export_logs_json(self, output_path: str, since: Optional[datetime] = None):
        """
        Export logs to JSON format.

        Args:
            output_path: Path for output JSON file
            since: Optional datetime to filter logs from
        """
        logs = []

        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_size > self.max_size * 10:
                continue  # Skip very large files

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if since:
                            # Parse timestamp and filter
                            try:
                                timestamp_str = line.split(' - ')[0]
                                log_time = datetime.strptime(
                                    timestamp_str, '%Y-%m-%d %H:%M:%S'
                                )
                                if log_time < since:
                                    continue
                            except (ValueError, IndexError):
                                continue

                        logs.append({
                            'file': log_file.name,
                            'line': line.strip()
                        })
            except Exception as e:
                self.log_error('export', f"Failed to read {log_file}: {e}")

        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=2)

        self.main_logger.info(f"Exported {len(logs)} log entries to {output_path}")

    def cleanup_old_logs(self, days: int = 30):
        """
        Remove log files older than specified days.

        Args:
            days: Number of days to keep logs
        """
        cutoff_time = datetime.now().timestamp() - (days * 86400)
        removed_count = 0

        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.log_error('cleanup', f"Failed to remove {log_file}: {e}")

        self.main_logger.info(f"Cleaned up {removed_count} old log files")


# Global logger instance
_logger_instance: Optional[DiagnosticLogger] = None


def get_logger() -> DiagnosticLogger:
    """Get or create global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DiagnosticLogger()
    return _logger_instance
