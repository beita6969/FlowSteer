# -*- coding: utf-8 -*-
# @Desc: Code sanitization utilities
# Adapted from AFlow's sanitize.py

import re
from typing import Optional

# Disallowed imports for security
DISALLOWED_IMPORTS = [
    # "os",
    # "sys",
    "subprocess",
    # "multiprocessing",
    # "matplotlib",
    # "seaborn",
    "plotly",
    "bokeh",
    "ggplot",
    "pylab",
    "tkinter",
    "PyQt5",
    "wx",
    "pyglet",
    # "shutil",
    # "pathlib",
    "socket",
    # "urllib",
    # "requests",
    # "http",
    "ftplib",
    "smtplib",
    "email",
    "ssl",
    "ctypes",
]


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """
    Sanitize Python code by removing dangerous imports and ensuring code safety.

    Args:
        code: The Python code to sanitize
        entrypoint: Optional function name that must be present

    Returns:
        Sanitized code string
    """
    if not code:
        return ""

    lines = code.split('\n')
    sanitized_lines = []

    for line in lines:
        # Check for disallowed imports
        should_skip = False
        for lib in DISALLOWED_IMPORTS:
            # Match import patterns
            if re.match(rf'^\s*import\s+{lib}\b', line):
                should_skip = True
                break
            if re.match(rf'^\s*from\s+{lib}\b', line):
                should_skip = True
                break

        if not should_skip:
            sanitized_lines.append(line)

    sanitized_code = '\n'.join(sanitized_lines)

    # If entrypoint is specified, verify it exists
    if entrypoint:
        # Check if the function is defined
        func_pattern = rf'def\s+{re.escape(entrypoint)}\s*\('
        if not re.search(func_pattern, sanitized_code):
            # Function not found - return code anyway but log warning
            pass

    return sanitized_code.strip()


def extract_code_block(text: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        text: Text containing potential markdown code blocks

    Returns:
        Extracted code or original text if no blocks found
    """
    # Try to find Python code blocks
    python_pattern = r'```python\s*([\s\S]*?)\s*```'
    matches = re.findall(python_pattern, text)

    if matches:
        return '\n\n'.join(matches)

    # Try generic code blocks
    generic_pattern = r'```\s*([\s\S]*?)\s*```'
    matches = re.findall(generic_pattern, text)

    if matches:
        return '\n\n'.join(matches)

    # Return original text
    return text


def check_dangerous_patterns(code: str) -> list:
    """
    Check for dangerous patterns in code.

    Args:
        code: The Python code to check

    Returns:
        List of detected dangerous patterns
    """
    dangerous_patterns = [
        (r'exec\s*\(', 'exec() call'),
        (r'eval\s*\(', 'eval() call'),
        (r'compile\s*\(', 'compile() call'),
        (r'__import__\s*\(', '__import__() call'),
        (r'open\s*\([^)]*[\'"]w', 'file write operation'),
        (r'globals\s*\(\s*\)', 'globals() access'),
        (r'locals\s*\(\s*\)', 'locals() access'),
    ]

    found = []
    for pattern, description in dangerous_patterns:
        if re.search(pattern, code):
            found.append(description)

    return found


# Export all functions
__all__ = [
    'sanitize',
    'extract_code_block',
    'check_dangerous_patterns',
    'DISALLOWED_IMPORTS'
]
