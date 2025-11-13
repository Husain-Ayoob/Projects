"""
Database utilities for AI Diagnostic Boot Drive.
Handles SQLite operations for diagnostics, fixes, and logging.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class DiagnosticDatabase:
    """SQLite database manager for diagnostic data."""

    def __init__(self, db_path: str = "/data/diagnostics.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create database tables if they don't exist."""

        # Diagnostics table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                hardware_info TEXT,
                software_info TEXT,
                issues_found TEXT,
                severity_summary TEXT,
                scan_duration REAL,
                status TEXT DEFAULT 'completed'
            )
        """)

        # Fixes table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fixes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnostic_id INTEGER,
                timestamp TEXT NOT NULL,
                issue_description TEXT NOT NULL,
                severity TEXT,
                script_executed TEXT NOT NULL,
                script_type TEXT,
                result TEXT,
                success BOOLEAN,
                duration REAL,
                error_message TEXT,
                FOREIGN KEY (diagnostic_id) REFERENCES diagnostics (id)
            )
        """)

        # System info table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_model TEXT,
                cpu_cores INTEGER,
                ram_total_gb REAL,
                storage_devices TEXT,
                gpu_info TEXT,
                os_info TEXT,
                boot_mode TEXT,
                network_interfaces TEXT
            )
        """)

        # Execution log table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                fix_id INTEGER,
                command TEXT NOT NULL,
                arguments TEXT,
                exit_code INTEGER,
                stdout TEXT,
                stderr TEXT,
                execution_time REAL,
                user_approved BOOLEAN,
                FOREIGN KEY (fix_id) REFERENCES fixes (id)
            )
        """)

        # AI analysis table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnostic_id INTEGER,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                tokens_used INTEGER,
                inference_time REAL,
                FOREIGN KEY (diagnostic_id) REFERENCES diagnostics (id)
            )
        """)

        self.conn.commit()

    def insert_diagnostic(self, scan_type: str, hardware_info: Dict,
                         software_info: Dict, issues_found: List[Dict],
                         severity_summary: Dict, scan_duration: float) -> int:
        """Insert a new diagnostic scan record."""
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO diagnostics
            (timestamp, scan_type, hardware_info, software_info,
             issues_found, severity_summary, scan_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            scan_type,
            json.dumps(hardware_info),
            json.dumps(software_info),
            json.dumps(issues_found),
            json.dumps(severity_summary),
            scan_duration
        ))

        self.conn.commit()
        return self.cursor.lastrowid

    def insert_fix(self, diagnostic_id: int, issue_description: str,
                   severity: str, script_executed: str, script_type: str,
                   result: str, success: bool, duration: float,
                   error_message: Optional[str] = None) -> int:
        """Insert a fix record."""
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO fixes
            (diagnostic_id, timestamp, issue_description, severity,
             script_executed, script_type, result, success, duration, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            diagnostic_id,
            timestamp,
            issue_description,
            severity,
            script_executed,
            script_type,
            result,
            success,
            duration,
            error_message
        ))

        self.conn.commit()
        return self.cursor.lastrowid

    def insert_system_info(self, system_data: Dict) -> int:
        """Insert system information record."""
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO system_info
            (timestamp, cpu_model, cpu_cores, ram_total_gb,
             storage_devices, gpu_info, os_info, boot_mode, network_interfaces)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            system_data.get('cpu_model'),
            system_data.get('cpu_cores'),
            system_data.get('ram_total_gb'),
            json.dumps(system_data.get('storage_devices', [])),
            json.dumps(system_data.get('gpu_info', {})),
            json.dumps(system_data.get('os_info', {})),
            system_data.get('boot_mode'),
            json.dumps(system_data.get('network_interfaces', []))
        ))

        self.conn.commit()
        return self.cursor.lastrowid

    def insert_execution_log(self, fix_id: int, command: str,
                            arguments: str, exit_code: int,
                            stdout: str, stderr: str, execution_time: float,
                            user_approved: bool) -> int:
        """Insert execution log entry."""
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO execution_log
            (timestamp, fix_id, command, arguments, exit_code,
             stdout, stderr, execution_time, user_approved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            fix_id,
            command,
            arguments,
            exit_code,
            stdout,
            stderr,
            execution_time,
            user_approved
        ))

        self.conn.commit()
        return self.cursor.lastrowid

    def insert_ai_analysis(self, diagnostic_id: int, prompt: str,
                          response: str, model_name: str,
                          tokens_used: int, inference_time: float) -> int:
        """Insert AI analysis record."""
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO ai_analysis
            (diagnostic_id, timestamp, prompt, response,
             model_name, tokens_used, inference_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            diagnostic_id,
            timestamp,
            prompt,
            response,
            model_name,
            tokens_used,
            inference_time
        ))

        self.conn.commit()
        return self.cursor.lastrowid

    def get_recent_diagnostics(self, limit: int = 10) -> List[Dict]:
        """Get recent diagnostic scans."""
        self.cursor.execute("""
            SELECT * FROM diagnostics
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_diagnostic_by_id(self, diagnostic_id: int) -> Optional[Dict]:
        """Get specific diagnostic by ID."""
        self.cursor.execute("""
            SELECT * FROM diagnostics WHERE id = ?
        """, (diagnostic_id,))

        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_fixes_for_diagnostic(self, diagnostic_id: int) -> List[Dict]:
        """Get all fixes for a specific diagnostic."""
        self.cursor.execute("""
            SELECT * FROM fixes
            WHERE diagnostic_id = ?
            ORDER BY timestamp DESC
        """, (diagnostic_id,))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_execution_logs(self, fix_id: int) -> List[Dict]:
        """Get execution logs for a specific fix."""
        self.cursor.execute("""
            SELECT * FROM execution_log
            WHERE fix_id = ?
            ORDER BY timestamp ASC
        """, (fix_id,))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {}

        # Total scans
        self.cursor.execute("SELECT COUNT(*) as count FROM diagnostics")
        stats['total_scans'] = self.cursor.fetchone()['count']

        # Total fixes
        self.cursor.execute("SELECT COUNT(*) as count FROM fixes")
        stats['total_fixes'] = self.cursor.fetchone()['count']

        # Successful fixes
        self.cursor.execute("SELECT COUNT(*) as count FROM fixes WHERE success = 1")
        stats['successful_fixes'] = self.cursor.fetchone()['count']

        # Failed fixes
        self.cursor.execute("SELECT COUNT(*) as count FROM fixes WHERE success = 0")
        stats['failed_fixes'] = self.cursor.fetchone()['count']

        # Average scan duration
        self.cursor.execute("SELECT AVG(scan_duration) as avg FROM diagnostics")
        stats['avg_scan_duration'] = self.cursor.fetchone()['avg'] or 0

        # Most recent scan
        self.cursor.execute("SELECT MAX(timestamp) as latest FROM diagnostics")
        stats['latest_scan'] = self.cursor.fetchone()['latest']

        return stats

    def export_to_json(self, output_path: str):
        """Export entire database to JSON."""
        data = {
            'diagnostics': self.get_recent_diagnostics(limit=1000),
            'statistics': self.get_statistics()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
