"""Path bootstrap so the dashboard modules and pages can import the project.

Streamlit runs ``dashboard/app.py`` (and every file under ``dashboard/pages/``)
as top-level scripts, putting ``dashboard/`` on ``sys.path`` but not the project
root. Call :func:`bootstrap` at the top of each page before importing project
packages (``config``, ``synth``, ...) or the sibling dashboard modules.
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent


def bootstrap() -> None:
    """Put the project root and the dashboard dir on ``sys.path`` (idempotent)."""
    for path in (PROJECT_ROOT, DASHBOARD_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
