"""
Execution safety module.
Provides safety checks, backups, and rollback capabilities.
"""

import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json


class ExecutionSafety:
    """Manages safe execution of fix scripts."""

    def __init__(self, backup_dir: str = "/data/backups",
                 max_backups: int = 10):
        """
        Initialize execution safety manager.

        Args:
            backup_dir: Directory for backups
            max_backups: Maximum number of backups to keep
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.backup_manifest = self.backup_dir / "manifest.json"
        self.backups = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load backup manifest."""
        if self.backup_manifest.exists():
            try:
                with open(self.backup_manifest, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {'backups': []}
        return {'backups': []}

    def _save_manifest(self):
        """Save backup manifest."""
        with open(self.backup_manifest, 'w') as f:
            json.dump(self.backups, f, indent=2)

    def create_backup(self, target_path: str, description: str = "") -> Optional[str]:
        """
        Create backup of a file or directory.

        Args:
            target_path: Path to backup
            description: Description of backup

        Returns:
            Backup path or None if failed
        """
        if not os.path.exists(target_path):
            return None

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        target_name = os.path.basename(target_path)
        backup_name = f"{target_name}.backup.{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            if os.path.isdir(target_path):
                shutil.copytree(target_path, backup_path)
            else:
                shutil.copy2(target_path, backup_path)

            # Add to manifest
            backup_info = {
                'timestamp': timestamp,
                'original_path': target_path,
                'backup_path': str(backup_path),
                'description': description,
                'size_bytes': self._get_size(backup_path)
            }

            self.backups['backups'].append(backup_info)
            self._save_manifest()

            # Cleanup old backups
            self._cleanup_old_backups()

            return str(backup_path)

        except Exception as e:
            print(f"Backup failed: {e}")
            return None

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore from backup.

        Args:
            backup_path: Path to backup file

        Returns:
            Success status
        """
        # Find backup in manifest
        backup_info = None
        for backup in self.backups['backups']:
            if backup['backup_path'] == backup_path:
                backup_info = backup
                break

        if not backup_info:
            return False

        original_path = backup_info['original_path']

        try:
            # Remove current file/directory
            if os.path.exists(original_path):
                if os.path.isdir(original_path):
                    shutil.rmtree(original_path)
                else:
                    os.remove(original_path)

            # Restore backup
            if os.path.isdir(backup_path):
                shutil.copytree(backup_path, original_path)
            else:
                shutil.copy2(backup_path, original_path)

            return True

        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    def _get_size(self, path: str) -> int:
        """Get total size of file or directory."""
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
            return total
        return 0

    def _cleanup_old_backups(self):
        """Remove old backups if exceeding max_backups."""
        if len(self.backups['backups']) > self.max_backups:
            # Sort by timestamp (oldest first)
            self.backups['backups'].sort(key=lambda x: x['timestamp'])

            # Remove oldest backups
            while len(self.backups['backups']) > self.max_backups:
                old_backup = self.backups['backups'].pop(0)
                backup_path = old_backup['backup_path']

                try:
                    if os.path.exists(backup_path):
                        if os.path.isdir(backup_path):
                            shutil.rmtree(backup_path)
                        else:
                            os.remove(backup_path)
                except Exception as e:
                    print(f"Failed to cleanup backup: {e}")

            self._save_manifest()

    def check_disk_space(self, required_mb: int = 100) -> Tuple[bool, int]:
        """
        Check if sufficient disk space is available.

        Args:
            required_mb: Required space in MB

        Returns:
            Tuple of (has_space, available_mb)
        """
        try:
            stat = shutil.disk_usage(self.backup_dir)
            available_mb = stat.free // (1024 * 1024)
            has_space = available_mb >= required_mb
            return has_space, available_mb
        except Exception:
            return False, 0

    def check_system_resources(self) -> Dict:
        """
        Check system resources before execution.

        Returns:
            Resource status dictionary
        """
        resources = {
            'safe_to_execute': True,
            'warnings': [],
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_space_mb': 0
        }

        # Check CPU usage
        try:
            result = subprocess.run(
                ['top', '-bn1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse CPU usage from top output
            for line in result.stdout.split('\n'):
                if 'Cpu(s)' in line:
                    # Extract CPU usage (simplified)
                    idle_match = line.split(',')
                    for part in idle_match:
                        if 'id' in part:
                            try:
                                idle = float(part.split()[0])
                                resources['cpu_usage'] = 100 - idle
                            except (ValueError, IndexError):
                                pass
                    break

            if resources['cpu_usage'] > 90:
                resources['warnings'].append('High CPU usage detected')

        except Exception:
            pass

        # Check memory usage
        try:
            result = subprocess.run(
                ['free', '-m'],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_line = lines[1].split()
                if len(mem_line) >= 3:
                    total = int(mem_line[1])
                    used = int(mem_line[2])
                    resources['memory_usage'] = (used / total) * 100

            if resources['memory_usage'] > 90:
                resources['warnings'].append('High memory usage detected')
                resources['safe_to_execute'] = False

        except Exception:
            pass

        # Check disk space
        has_space, available_mb = self.check_disk_space(100)
        resources['disk_space_mb'] = available_mb

        if not has_space:
            resources['warnings'].append('Low disk space')
            resources['safe_to_execute'] = False

        return resources

    def create_restore_point(self, description: str,
                            paths_to_backup: List[str]) -> Optional[str]:
        """
        Create a restore point with multiple backups.

        Args:
            description: Description of restore point
            paths_to_backup: List of paths to backup

        Returns:
            Restore point ID or None if failed
        """
        restore_point_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        restore_point = {
            'id': restore_point_id,
            'timestamp': restore_point_id,
            'description': description,
            'backups': []
        }

        for path in paths_to_backup:
            backup_path = self.create_backup(path, f"Restore point: {description}")
            if backup_path:
                restore_point['backups'].append({
                    'original': path,
                    'backup': backup_path
                })

        if restore_point['backups']:
            # Save restore point info
            restore_point_file = self.backup_dir / f"restore_point_{restore_point_id}.json"
            with open(restore_point_file, 'w') as f:
                json.dump(restore_point, f, indent=2)

            return restore_point_id

        return None

    def restore_point(self, restore_point_id: str) -> bool:
        """
        Restore from a restore point.

        Args:
            restore_point_id: Restore point identifier

        Returns:
            Success status
        """
        restore_point_file = self.backup_dir / f"restore_point_{restore_point_id}.json"

        if not restore_point_file.exists():
            return False

        try:
            with open(restore_point_file, 'r') as f:
                restore_point = json.load(f)

            # Restore all backups
            for backup_info in restore_point['backups']:
                backup_path = backup_info['backup']
                if not self.restore_backup(backup_path):
                    # Partial restore failure
                    return False

            return True

        except Exception as e:
            print(f"Restore point restoration failed: {e}")
            return False

    def list_restore_points(self) -> List[Dict]:
        """List all available restore points."""
        restore_points = []

        for rp_file in self.backup_dir.glob("restore_point_*.json"):
            try:
                with open(rp_file, 'r') as f:
                    restore_point = json.load(f)
                    restore_points.append({
                        'id': restore_point['id'],
                        'timestamp': restore_point['timestamp'],
                        'description': restore_point['description'],
                        'backup_count': len(restore_point['backups'])
                    })
            except Exception:
                continue

        return sorted(restore_points, key=lambda x: x['timestamp'], reverse=True)

    def verify_execution_safety(self, script: str, risk_level: str) -> Tuple[bool, List[str]]:
        """
        Verify if it's safe to execute a script.

        Args:
            script: Script content
            risk_level: Risk level (low, medium, high)

        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True

        # Check system resources
        resources = self.check_system_resources()
        if not resources['safe_to_execute']:
            is_safe = False
            warnings.extend(resources['warnings'])

        # Additional checks based on risk level
        if risk_level == 'high':
            warnings.append('High-risk operation - manual review required')
            is_safe = False  # Require explicit approval

        elif risk_level == 'medium':
            warnings.append('Medium-risk operation - backup recommended')

        return is_safe, warnings
