"""
main.py
-------
Application entry point for the Car Parking Space Counter.

Usage
-----
  python main.py                        # Launch the GUI
  python main.py --config path/to/cfg   # Use a custom config file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car Parking Space Counter — CV Project"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)

    # Import Qt application here (after argument parsing)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon
    from ui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Parking Space Counter")
    app.setStyle("Fusion")

    window = MainWindow(config_path=str(config_path))
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
