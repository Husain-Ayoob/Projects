"""
Software diagnostics module.
Performs software health checks, log analysis, and system integrity verification.
"""

import subprocess
import re
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SoftwareIssue:
    """Represents a software issue."""
    component: str
    severity: str  # critical, high, medium, low
    issue: str
    details: Dict
    recommendation: str
    log_excerpt: str = ""


class SoftwareDiagnostics:
    """Software diagnostic scanner."""

    def __init__(self, timeout: int = 60):
        """
        Initialize software diagnostics.

        Args:
            timeout: Timeout for each diagnostic command in seconds
        """
        self.timeout = timeout
        self.issues: List[SoftwareIssue] = []

    def run_command(self, command: List[str]) -> Tuple[str, str, int]:
        """
        Run a system command safely.

        Args:
            command: Command and arguments as list

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", -1
        except FileNotFoundError:
            return "", f"Command not found: {command[0]}", -1
        except Exception as e:
            return "", str(e), -1

    def check_bootloader(self) -> Dict:
        """Check bootloader status."""
        bootloader_info = {
            'type': 'unknown',
            'status': 'unknown',
            'boot_mode': 'unknown',
            'issues': []
        }

        # Detect boot mode (UEFI or BIOS)
        if os.path.exists('/sys/firmware/efi'):
            bootloader_info['boot_mode'] = 'UEFI'
        else:
            bootloader_info['boot_mode'] = 'BIOS/Legacy'

        # Check for GRUB
        if os.path.exists('/boot/grub/grub.cfg') or os.path.exists('/boot/grub2/grub.cfg'):
            bootloader_info['type'] = 'GRUB'
            bootloader_info['status'] = 'installed'

            # Check GRUB configuration
            grub_cfg = '/boot/grub/grub.cfg' if os.path.exists('/boot/grub/grub.cfg') else '/boot/grub2/grub.cfg'
            try:
                with open(grub_cfg, 'r') as f:
                    grub_content = f.read()
                    if not grub_content.strip():
                        self.issues.append(SoftwareIssue(
                            component='Bootloader',
                            severity='critical',
                            issue='GRUB configuration file is empty',
                            details={'config_file': grub_cfg},
                            recommendation='Regenerate GRUB configuration with grub-mkconfig'
                        ))
            except Exception as e:
                self.issues.append(SoftwareIssue(
                    component='Bootloader',
                    severity='high',
                    issue=f'Cannot read GRUB configuration: {str(e)}',
                    details={'error': str(e)},
                    recommendation='Check file permissions and integrity'
                ))

        # Check for systemd-boot
        elif os.path.exists('/boot/loader/loader.conf'):
            bootloader_info['type'] = 'systemd-boot'
            bootloader_info['status'] = 'installed'

        return bootloader_info

    def check_filesystems(self) -> Dict:
        """Check filesystem integrity."""
        fs_info = {
            'mounted': [],
            'issues': []
        }

        # Get mounted filesystems
        stdout, _, returncode = self.run_command(['mount'])
        if returncode == 0:
            for line in stdout.split('\n'):
                if line.strip() and ' on ' in line:
                    parts = line.split(' on ')
                    if len(parts) >= 2:
                        device = parts[0]
                        mount_parts = parts[1].split(' type ')
                        if len(mount_parts) >= 2:
                            mountpoint = mount_parts[0]
                            fs_type = mount_parts[1].split()[0]

                            fs_info['mounted'].append({
                                'device': device,
                                'mountpoint': mountpoint,
                                'type': fs_type
                            })

        # Check disk usage
        stdout, _, returncode = self.run_command(['df', '-h'])
        if returncode == 0:
            for line in stdout.split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    usage_str = parts[4].rstrip('%')
                    try:
                        usage = int(usage_str)
                        mountpoint = parts[5] if len(parts) > 5 else parts[4]

                        if usage >= 95:
                            self.issues.append(SoftwareIssue(
                                component='Filesystem',
                                severity='critical',
                                issue=f'Filesystem {mountpoint} is {usage}% full',
                                details={'mountpoint': mountpoint, 'usage': usage},
                                recommendation='Free up disk space immediately'
                            ))
                        elif usage >= 85:
                            self.issues.append(SoftwareIssue(
                                component='Filesystem',
                                severity='medium',
                                issue=f'Filesystem {mountpoint} is {usage}% full',
                                details={'mountpoint': mountpoint, 'usage': usage},
                                recommendation='Consider freeing up disk space'
                            ))
                    except ValueError:
                        continue

        return fs_info

    def analyze_system_logs(self) -> Dict:
        """Analyze system logs for errors and warnings."""
        log_info = {
            'errors_found': 0,
            'warnings_found': 0,
            'critical_found': 0,
            'recent_errors': [],
            'issues': []
        }

        # Analyze journalctl logs (systemd)
        stdout, _, returncode = self.run_command([
            'journalctl', '-p', 'err', '-n', '50', '--no-pager'
        ])

        if returncode == 0:
            error_lines = [line for line in stdout.split('\n') if line.strip()]
            log_info['errors_found'] = len(error_lines)
            log_info['recent_errors'] = error_lines[:10]  # Store first 10

            # Look for critical patterns
            critical_patterns = [
                (r'kernel.*panic', 'Kernel panic detected'),
                (r'Out of memory', 'Out of memory error'),
                (r'I/O error', 'I/O error detected'),
                (r'failed.*mount', 'Filesystem mount failure'),
                (r'segfault', 'Segmentation fault detected'),
                (r'hardware error', 'Hardware error reported'),
            ]

            for pattern, description in critical_patterns:
                matches = [line for line in error_lines if re.search(pattern, line, re.IGNORECASE)]
                if matches:
                    log_info['critical_found'] += len(matches)
                    self.issues.append(SoftwareIssue(
                        component='System Logs',
                        severity='high',
                        issue=description,
                        details={'count': len(matches)},
                        recommendation='Investigate system logs for root cause',
                        log_excerpt=matches[0][:200]
                    ))

        # Check for failed services
        stdout, _, returncode = self.run_command(['systemctl', '--failed', '--no-pager'])
        if returncode == 0:
            failed_services = [
                line.strip() for line in stdout.split('\n')
                if line.strip() and '.service' in line
            ]
            if failed_services:
                self.issues.append(SoftwareIssue(
                    component='Services',
                    severity='medium',
                    issue=f'{len(failed_services)} failed services detected',
                    details={'failed_services': failed_services[:5]},
                    recommendation='Check service status and logs'
                ))

        return log_info

    def check_processes(self) -> Dict:
        """Check running processes and resource usage."""
        process_info = {
            'total': 0,
            'zombie': 0,
            'high_cpu': [],
            'high_memory': [],
            'issues': []
        }

        # Get process count
        stdout, _, returncode = self.run_command(['ps', 'aux'])
        if returncode == 0:
            process_lines = stdout.strip().split('\n')[1:]  # Skip header
            process_info['total'] = len(process_lines)

            # Check for zombie processes
            for line in process_lines:
                if '<defunct>' in line or ' Z ' in line:
                    process_info['zombie'] += 1

            # Parse high resource usage processes
            for line in process_lines:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    try:
                        cpu = float(parts[2])
                        mem = float(parts[3])
                        command = parts[10]

                        if cpu > 80:
                            process_info['high_cpu'].append({
                                'command': command[:50],
                                'cpu': cpu
                            })

                        if mem > 20:
                            process_info['high_memory'].append({
                                'command': command[:50],
                                'memory': mem
                            })
                    except (ValueError, IndexError):
                        continue

        # Report zombie processes
        if process_info['zombie'] > 5:
            self.issues.append(SoftwareIssue(
                component='Processes',
                severity='medium',
                issue=f'{process_info["zombie"]} zombie processes detected',
                details={'count': process_info['zombie']},
                recommendation='Investigate parent processes and system state'
            ))

        return process_info

    def check_boot_issues(self) -> Dict:
        """Check for boot-related issues."""
        boot_info = {
            'last_boot': None,
            'boot_errors': [],
            'issues': []
        }

        # Get last boot time
        stdout, _, returncode = self.run_command(['uptime', '-s'])
        if returncode == 0:
            boot_info['last_boot'] = stdout.strip()

        # Check dmesg for boot errors
        stdout, _, returncode = self.run_command(['dmesg', '--level=err,warn', '-H'])
        if returncode == 0:
            error_lines = [line for line in stdout.split('\n') if line.strip()]
            boot_info['boot_errors'] = error_lines[:20]

            # Look for specific boot issues
            boot_patterns = [
                (r'failed to load', 'Failed to load driver or module'),
                (r'not found', 'Missing driver or firmware'),
                (r'timeout', 'Device timeout during boot'),
            ]

            for pattern, description in boot_patterns:
                matches = [line for line in error_lines if re.search(pattern, line, re.IGNORECASE)]
                if matches:
                    self.issues.append(SoftwareIssue(
                        component='Boot',
                        severity='medium',
                        issue=description,
                        details={'count': len(matches)},
                        recommendation='Check dmesg output for details',
                        log_excerpt=matches[0][:200]
                    ))

        return boot_info

    def check_malware_signatures(self) -> Dict:
        """Quick malware signature check (basic)."""
        malware_info = {
            'scan_performed': False,
            'threats_found': 0,
            'issues': []
        }

        # Check if ClamAV is available
        stdout, _, returncode = self.run_command(['which', 'clamscan'])
        if returncode != 0:
            malware_info['scan_performed'] = False
            return malware_info

        # Note: Full scan would take too long, skip for MVP
        # This is a placeholder for future implementation
        malware_info['scan_performed'] = False

        return malware_info

    def run_full_scan(self) -> Dict:
        """
        Run complete software diagnostic scan.

        Returns:
            Dictionary with all software information
        """
        self.issues = []  # Reset issues

        results = {
            'bootloader': self.check_bootloader(),
            'filesystems': self.check_filesystems(),
            'logs': self.analyze_system_logs(),
            'processes': self.check_processes(),
            'boot': self.check_boot_issues(),
            'malware': self.check_malware_signatures(),
            'issues': [asdict(issue) for issue in self.issues]
        }

        return results

    def get_severity_summary(self) -> Dict:
        """Get summary of issues by severity."""
        summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'total': len(self.issues)
        }

        for issue in self.issues:
            severity = issue.severity.lower()
            if severity in summary:
                summary[severity] += 1

        return summary
