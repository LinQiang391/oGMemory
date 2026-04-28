"""Interactive prompt helpers for CLI wizards."""

from __future__ import annotations

import sys
from getpass import getpass


def confirm(msg: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns default when not a TTY."""
    if not sys.stdin.isatty():
        return default
    prompt = f"{msg} [Y/n]" if default else f"{msg} [y/N]"
    response = input(f"  {prompt} ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def select(msg: str, choices: list[str], default: int = 0) -> int:
    """Ask user to select from a list. Returns default when not a TTY."""
    if not sys.stdin.isatty():
        return default
    print(f"  {msg}")
    for i, choice in enumerate(choices, 1):
        print(f"    {i}) {choice}")
    response = input(f"  Choice [{default + 1}]: ").strip()
    if not response:
        return default
    try:
        idx = int(response) - 1
        return max(0, min(idx, len(choices) - 1))
    except ValueError:
        return default


def input_validated(msg: str, default: str = "", optional: bool = False) -> str:
    """Get input with optional default and validation. Returns default when not a TTY."""
    if not sys.stdin.isatty():
        return default
    prompt = f"{msg} [{default}]" if default else msg
    response = input(f"  {prompt}: ").strip()
    if not response:
        if not optional and not default:
            return input_validated(msg, default, optional)
        return default
    return response


def input_masked(msg: str) -> str:
    """Get password input without echo. Returns empty string when not a TTY."""
    if not sys.stdin.isatty():
        return ""
    return getpass(f"  {msg}: ")
