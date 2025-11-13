# AI Diagnostic Boot Drive - MVP

A bootable USB/hard drive diagnostic system with embedded LLM that autonomously detects, diagnoses, and fixes computer issues through AI-powered code generation and execution.

## Overview

The AI Diagnostic Boot Drive is an intelligent system diagnostic tool that combines traditional hardware/software diagnostics with AI-powered analysis and automated fix generation. It boots independently of the host system and can diagnose and repair issues even when the primary OS is non-functional.

## Features

### Core Features (MVP)
- ✅ **Bootable Environment**: UEFI and Legacy BIOS compatible
- ✅ **Hardware Diagnostics**: CPU, RAM, Storage (SMART), Network, GPU detection
- ✅ **Software Diagnostics**: Bootloader, filesystem, logs, processes
- ✅ **AI Analysis**: Llama 3.2 3B powered issue analysis
- ✅ **Automated Fix Generation**: AI-generated bash/Python scripts
- ✅ **Safety Features**: Script validation, backups, rollback capability
- ✅ **Terminal UI**: Color-coded, interactive command-line interface
- ✅ **Report Generation**: Text, JSON, and HTML reports
- ✅ **SQLite Logging**: Complete audit trail of all operations

### Safety Features
- Command whitelist/blacklist
- Syntax validation before execution
- Automatic backups before destructive operations
- Dry-run mode
- User approval required for high-risk operations
- Execution timeout protection
- Rollback capability

## System Architecture

```
┌─────────────────────────────────────┐
│   Bootable Linux (Debian Live)     │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐  │
│  │   Diagnostic Orchestrator    │  │
│  │      (main.py)               │  │
│  └──────────────────────────────┘  │
│           │          │               │
│     ┌─────▼────┐  ┌─▼────────┐     │
│     │ Hardware │  │ Software │     │
│     │ Scanner  │  │ Scanner  │     │
│     └─────┬────┘  └─┬────────┘     │
│           │          │               │
│     ┌─────▼──────────▼────┐         │
│     │   LLM Engine        │         │
│     │   (llama.cpp)       │         │
│     └─────────┬───────────┘         │
│               │                      │
│     ┌─────────▼───────────┐         │
│     │  Code Generator &   │         │
│     │  Validator          │         │
│     └─────────┬───────────┘         │
│               │                      │
│     ┌─────────▼───────────┐         │
│     │  Execution Engine   │         │
│     │  (Sandboxed)        │         │
│     └─────────────────────┘         │
└─────────────────────────────────────┘
```

## Installation & Setup

### Prerequisites
- USB drive or hard drive (8GB minimum)
- Debian Live build tools
- Python 3.11+
- llama.cpp
- Llama 3.2 3B model (quantized to 4-bit, ~2GB)

### Quick Start (Development)

1. **Clone the repository**
```bash
cd ai-diagnostic-boot
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Download LLM model**
```bash
# Download Llama 3.2 3B quantized model
# Place in system/model.gguf
wget https://huggingface.co/TheBloke/Llama-3.2-3B-GGUF/resolve/main/llama-3.2-3b.Q4_K_M.gguf -O system/model.gguf
```

4. **Run diagnostic system (requires root)**
```bash
sudo python3 src/main.py
```

### Building Bootable Image (Future)

Instructions for creating bootable USB/ISO will be added in future releases using Debian Live build system.

## Usage

### Main Menu

```
================================
AI Diagnostic Boot Drive v1.0
================================
1. Run Full System Diagnostics
2. Quick Hardware Scan
3. Fix Detected Issues (Auto)
4. View Diagnostic Report
5. Manual Repair Mode
6. View Logs
7. Exit to Shell
================================
```

### Workflow

1. **Boot from USB/Drive**
   - System boots into diagnostic environment
   - Automatic hardware detection

2. **Run Diagnostics**
   - Select "Run Full System Diagnostics"
   - System scans hardware and software
   - Results displayed with severity levels

3. **AI Analysis**
   - AI analyzes detected issues
   - Generates fix scripts automatically
   - Displays recommended fixes

4. **Review & Execute Fixes**
   - Review generated scripts
   - Approve execution
   - Monitor progress
   - View results

5. **Generate Report**
   - Export diagnostic report (Text/JSON/HTML)
   - Save for records

## Configuration

### settings.json

Located in `config/settings.json`:

```json
{
  "system": {
    "version": "1.0.0",
    "mode": "safe",
    "max_execution_time": 300
  },
  "llm": {
    "model_path": "/system/model.gguf",
    "context_window": 8192,
    "temperature": 0.3
  },
  "execution": {
    "dry_run_first": true,
    "require_user_approval": true
  }
}
```

### whitelist.json

Command whitelist/blacklist configuration in `config/whitelist.json`

## Project Structure

```
ai-diagnostic-boot/
├── boot/
│   ├── grub.cfg              # Boot configuration
│   └── kernel-params         # Kernel parameters
├── system/
│   ├── model.gguf            # LLM model (to be added)
│   └── system-prompt.txt     # AI system prompt
├── src/
│   ├── main.py               # Main orchestrator
│   ├── diagnostics/
│   │   ├── hardware.py       # Hardware scanning
│   │   ├── software.py       # Software analysis
│   │   └── parsers.py        # Data parsing
│   ├── ai/
│   │   ├── llm_interface.py  # LLM communication
│   │   ├── code_gen.py       # Code generation
│   │   └── validator.py      # Code validation
│   ├── execution/
│   │   ├── runner.py         # Script execution
│   │   └── safety.py         # Safety checks
│   ├── ui/
│   │   ├── terminal.py       # Terminal UI
│   │   └── reports.py        # Report generation
│   └── utils/
│       ├── logging.py        # Logging utilities
│       └── database.py       # SQLite operations
├── config/
│   ├── settings.json         # System configuration
│   └── whitelist.json        # Command whitelist
├── data/
│   ├── diagnostics.db        # SQLite database
│   └── logs/                 # Log files
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Diagnostic Capabilities

### Hardware Checks
- **CPU**: Model, cores, temperature, usage
- **RAM**: Capacity, usage, swap, memory testing
- **Storage**: SMART data, bad sectors, filesystem integrity
- **Network**: Interface detection, connectivity
- **GPU**: Detection and basic info

### Software Checks
- **Bootloader**: GRUB/systemd-boot status
- **Filesystems**: Mount status, disk usage, corruption
- **System Logs**: Error analysis (journalctl, dmesg)
- **Processes**: Running processes, zombies, resource usage
- **Boot Issues**: Boot errors and warnings

## Safety & Security

### Execution Safety
- All commands validated against whitelist
- Forbidden patterns blocked (rm -rf /, dd, mkfs, etc.)
- Syntax checking before execution
- Automatic backups before changes
- Rollback capability on failure
- Execution timeout protection

### Data Protection
- Read-only mode available
- No data collection or external transmission
- All operations logged locally
- Backup creation before destructive operations

## Limitations (MVP)

- No Windows-specific diagnostics (BSOD, Registry)
- No advanced malware scanning (ClamAV basic check only)
- No remote diagnostics capability
- Terminal UI only (no web interface)
- English language only

## Future Enhancements

- Web-based UI (Flask app)
- Windows diagnostics support
- Network diagnostics and repair
- Driver issue detection
- Cloud sync for diagnostic history
- Multi-language support
- Voice-guided diagnostics
- Remote diagnostics via internet
- Integration with ticketing systems

## Development

### Running Tests
```bash
# Unit tests (to be implemented)
python -m pytest tests/
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Troubleshooting

### LLM Model Not Found
```bash
# Ensure model is in correct location
ls -lh system/model.gguf
```

### Permission Denied
```bash
# Run with sudo
sudo python3 src/main.py
```

### Hardware Tools Missing
```bash
# Install required tools (Debian/Ubuntu)
sudo apt-get install smartmontools lm-sensors memtester
```

## License

[To be determined]

## Credits

- **LLM**: Meta Llama 3.2 3B
- **Runtime**: llama.cpp
- **Base OS**: Debian Live

## Support

For issues and questions, please create an issue in the repository.

---

**Note**: This is an MVP (Minimum Viable Product). Features and functionality will be expanded in future releases.
