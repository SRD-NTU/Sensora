"""Run with: streamlit run app.py"""
import sys
from pathlib import Path

# Use this project's source tree, not any other editable install of sensora.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sensora.optimizer_ui import run  # noqa: E402

run()
