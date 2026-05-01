"""
Small helper utilities shared across modules.
"""

import os


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def print_section(number, title):
    """Print a numbered section header."""
    print(f"\n{'=' * 60}")
    print(f"[{number}] {title}")
    print("=" * 60)
