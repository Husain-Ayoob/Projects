"""
Hardware diagnostics module.
Performs comprehensive hardware scanning and health checks.
"""

import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class HardwareIssue:
    """Represents a hardware issue."""
    component: str
    severity: str  # critical, high, medium, low
    issue: str
    details: Dict
    recommendation: str


class HardwareDiagnostics:
    """Hardware diagnostic scanner."""

    def __init__(self, timeout: int = 60):
        """
        Initialize hardware diagnostics.

        Args:
            timeout: Timeout for each diagnostic command in seconds
        """
        self.timeout = timeout
        self.issues: List[HardwareIssue] = []

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

    def scan_cpu(self) -> Dict:
        """Scan CPU information and health."""
        cpu_info = {
            'model': 'Unknown',
            'cores': 0,
            'threads': 0,
            'temperature': [],
            'usage': 0,
            'issues': []
        }

        # Get CPU info from lscpu
        stdout, _, returncode = self.run_command(['lscpu'])
        if returncode == 0:
            for line in stdout.split('\n'):
                if 'Model name:' in line:
                    cpu_info['model'] = line.split(':', 1)[1].strip()
                elif 'CPU(s):' in line and 'NUMA' not in line:
                    try:
                        cpu_info['threads'] = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif 'Core(s) per socket:' in line:
                    try:
                        cpu_info['cores'] = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass

        # Get CPU temperature (if available)
        stdout, _, returncode = self.run_command(['sensors'])
        if returncode == 0:
            temp_pattern = re.compile(r'Core \d+:\s+\+(\d+\.\d+)°C')
            temperatures = temp_pattern.findall(stdout)
            cpu_info['temperature'] = [float(t) for t in temperatures]

            # Check for overheating
            if temperatures:
                max_temp = max(float(t) for t in temperatures)
                if max_temp > 90:
                    self.issues.append(HardwareIssue(
                        component='CPU',
                        severity='critical',
                        issue=f'CPU overheating: {max_temp}°C',
                        details={'max_temperature': max_temp, 'all_temps': cpu_info['temperature']},
                        recommendation='Check CPU cooling system and thermal paste'
                    ))
                elif max_temp > 80:
                    self.issues.append(HardwareIssue(
                        component='CPU',
                        severity='high',
                        issue=f'CPU running hot: {max_temp}°C',
                        details={'max_temperature': max_temp},
                        recommendation='Monitor CPU temperature and improve cooling'
                    ))

        return cpu_info

    def scan_memory(self) -> Dict:
        """Scan RAM information and perform basic checks."""
        mem_info = {
            'total_gb': 0,
            'available_gb': 0,
            'used_gb': 0,
            'usage_percent': 0,
            'swap_total_gb': 0,
            'swap_used_gb': 0,
            'issues': []
        }

        # Get memory info from free command
        stdout, _, returncode = self.run_command(['free', '-m'])
        if returncode == 0:
            lines = stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_line = lines[1].split()
                if len(mem_line) >= 7:
                    total_mb = int(mem_line[1])
                    used_mb = int(mem_line[2])
                    available_mb = int(mem_line[6]) if len(mem_line) > 6 else 0

                    mem_info['total_gb'] = round(total_mb / 1024, 2)
                    mem_info['used_gb'] = round(used_mb / 1024, 2)
                    mem_info['available_gb'] = round(available_mb / 1024, 2)
                    mem_info['usage_percent'] = round((used_mb / total_mb) * 100, 1)

            if len(lines) >= 3:
                swap_line = lines[2].split()
                if len(swap_line) >= 3:
                    swap_total_mb = int(swap_line[1])
                    swap_used_mb = int(swap_line[2])
                    mem_info['swap_total_gb'] = round(swap_total_mb / 1024, 2)
                    mem_info['swap_used_gb'] = round(swap_used_mb / 1024, 2)

        # Check for memory issues
        if mem_info['usage_percent'] > 95:
            self.issues.append(HardwareIssue(
                component='RAM',
                severity='critical',
                issue=f'Memory critically low: {mem_info["usage_percent"]}% used',
                details={'usage_percent': mem_info['usage_percent']},
                recommendation='Close applications or add more RAM'
            ))
        elif mem_info['usage_percent'] > 85:
            self.issues.append(HardwareIssue(
                component='RAM',
                severity='medium',
                issue=f'High memory usage: {mem_info["usage_percent"]}% used',
                details={'usage_percent': mem_info['usage_percent']},
                recommendation='Monitor memory usage'
            ))

        # Check swap usage
        if mem_info['swap_total_gb'] > 0:
            swap_usage_percent = (mem_info['swap_used_gb'] / mem_info['swap_total_gb']) * 100
            if swap_usage_percent > 50:
                self.issues.append(HardwareIssue(
                    component='RAM',
                    severity='high',
                    issue=f'Heavy swap usage: {swap_usage_percent:.1f}%',
                    details={'swap_usage_percent': swap_usage_percent},
                    recommendation='System may be low on RAM, consider adding more'
                ))

        return mem_info

    def scan_storage(self) -> Dict:
        """Scan storage devices and check health."""
        storage_info = {
            'devices': [],
            'issues': []
        }

        # Get block device info
        stdout, _, returncode = self.run_command(['lsblk', '-J', '-o', 'NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE'])
        if returncode == 0:
            try:
                lsblk_data = json.loads(stdout)
                storage_info['devices'] = lsblk_data.get('blockdevices', [])
            except json.JSONDecodeError:
                pass

        # Get SMART data for each disk
        for device in storage_info['devices']:
            if device.get('type') == 'disk':
                device_name = device.get('name')
                smart_data = self._check_smart_data(f"/dev/{device_name}")
                device['smart'] = smart_data

        return storage_info

    def _check_smart_data(self, device: str) -> Dict:
        """
        Check SMART data for a storage device.

        Args:
            device: Device path (e.g., /dev/sda)

        Returns:
            Dictionary with SMART data
        """
        smart_info = {
            'available': False,
            'healthy': True,
            'temperature': None,
            'power_on_hours': None,
            'issues': []
        }

        # Check if smartctl is available
        stdout, stderr, returncode = self.run_command(['smartctl', '-H', device])

        if "command not found" in stderr or returncode == 127:
            return smart_info

        smart_info['available'] = True

        # Check overall health
        if returncode != 0 or 'PASSED' not in stdout:
            smart_info['healthy'] = False
            self.issues.append(HardwareIssue(
                component='Storage',
                severity='critical',
                issue=f'SMART health check failed for {device}',
                details={'device': device},
                recommendation='Backup data immediately and replace drive'
            ))

        # Get detailed attributes
        stdout, _, returncode = self.run_command(['smartctl', '-A', device])
        if returncode == 0:
            # Parse temperature
            temp_match = re.search(r'Temperature_Celsius.*\s+(\d+)', stdout)
            if temp_match:
                temp = int(temp_match.group(1))
                smart_info['temperature'] = temp
                if temp > 60:
                    self.issues.append(HardwareIssue(
                        component='Storage',
                        severity='medium',
                        issue=f'Drive {device} running hot: {temp}°C',
                        details={'device': device, 'temperature': temp},
                        recommendation='Improve drive cooling'
                    ))

            # Parse power-on hours
            hours_match = re.search(r'Power_On_Hours.*\s+(\d+)', stdout)
            if hours_match:
                smart_info['power_on_hours'] = int(hours_match.group(1))

            # Check for reallocated sectors
            realloc_match = re.search(r'Reallocated_Sector_Ct.*\s+(\d+)', stdout)
            if realloc_match:
                realloc_count = int(realloc_match.group(1))
                if realloc_count > 0:
                    self.issues.append(HardwareIssue(
                        component='Storage',
                        severity='high',
                        issue=f'Drive {device} has {realloc_count} reallocated sectors',
                        details={'device': device, 'reallocated_sectors': realloc_count},
                        recommendation='Backup data and monitor drive health'
                    ))

        return smart_info

    def scan_network(self) -> Dict:
        """Scan network interfaces."""
        network_info = {
            'interfaces': [],
            'issues': []
        }

        # Get network interfaces
        stdout, _, returncode = self.run_command(['ip', '-j', 'addr'])
        if returncode == 0:
            try:
                interfaces = json.loads(stdout)
                network_info['interfaces'] = [
                    {
                        'name': iface.get('ifname'),
                        'state': iface.get('operstate'),
                        'addresses': [
                            addr.get('local') for addr in iface.get('addr_info', [])
                        ]
                    }
                    for iface in interfaces
                ]
            except json.JSONDecodeError:
                pass

        # Check for interfaces that are down
        for iface in network_info['interfaces']:
            if iface['name'] != 'lo' and iface['state'] != 'UP':
                self.issues.append(HardwareIssue(
                    component='Network',
                    severity='low',
                    issue=f'Network interface {iface["name"]} is down',
                    details={'interface': iface['name'], 'state': iface['state']},
                    recommendation='Check network cable or wireless connection'
                ))

        return network_info

    def scan_gpu(self) -> Dict:
        """Scan GPU information."""
        gpu_info = {
            'detected': False,
            'devices': [],
            'issues': []
        }

        # Check for GPU using lspci
        stdout, _, returncode = self.run_command(['lspci'])
        if returncode == 0:
            gpu_lines = [
                line for line in stdout.split('\n')
                if 'VGA' in line or 'Display' in line or '3D' in line
            ]

            if gpu_lines:
                gpu_info['detected'] = True
                gpu_info['devices'] = gpu_lines

        return gpu_info

    def run_full_scan(self) -> Dict:
        """
        Run complete hardware diagnostic scan.

        Returns:
            Dictionary with all hardware information
        """
        self.issues = []  # Reset issues

        results = {
            'cpu': self.scan_cpu(),
            'memory': self.scan_memory(),
            'storage': self.scan_storage(),
            'network': self.scan_network(),
            'gpu': self.scan_gpu(),
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
