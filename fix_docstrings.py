#!/usr/bin/env python3
"""Fix docstring lines to be max 72 characters."""

import re
from pathlib import Path


def wrap_line(line, max_length=72, indent_level=0):
    """Wrap a line to max_length, preserving indentation."""
    indent = " " * indent_level
    stripped = line.strip()

    # If it's already short enough, return it
    if len(indent + stripped) <= max_length:
        return [indent + stripped]

    # For parameter descriptions and similar, preserve structure
    if ":" in stripped[:30]:  # Parameter, Returns, etc.
        # Find the colon position
        colon_pos = stripped.index(":")
        prefix = stripped[: colon_pos + 1]
        rest = stripped[colon_pos + 1 :].strip()

        if len(indent + prefix + " " + rest) <= max_length:
            return [indent + prefix + " " + rest]

        # Need to wrap the description part
        result = [indent + prefix]
        # Add extra indent for continuation
        cont_indent = indent + "    "
        words = rest.split()
        current_line = cont_indent

        for word in words:
            if len(current_line + " " + word) <= max_length:
                if current_line == cont_indent:
                    current_line += word
                else:
                    current_line += " " + word
            else:
                if current_line != cont_indent:
                    result.append(current_line)
                current_line = cont_indent + word

        if current_line != cont_indent:
            result.append(current_line)

        return result

    # Regular wrapping
    words = stripped.split()
    lines = []
    current_line = indent

    for word in words:
        if len(current_line + " " + word) <= max_length:
            if current_line == indent:
                current_line += word
            else:
                current_line += " " + word
        else:
            if current_line != indent:
                lines.append(current_line)
            current_line = indent + word

    if current_line != indent:
        lines.append(current_line)

    return lines if lines else [indent + stripped]


def fix_docstrings_in_file(filepath):
    """Fix docstrings in a single file."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    result = []
    in_docstring = False
    docstring_delimiter = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if we're entering/exiting a docstring
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_delimiter = '"""' if '"""' in line else "'''"
                # Check if it's a one-line docstring
                if line.count(docstring_delimiter) == 2:
                    in_docstring = False
                    docstring_delimiter = None
            elif docstring_delimiter in line:
                in_docstring = False
                docstring_delimiter = None

        # Process the line
        if in_docstring and len(line.rstrip()) > 72:
            # Determine indentation
            indent_match = re.match(r"^(\s*)", line)
            indent_level = len(indent_match.group(1)) if indent_match else 0

            # Wrap the line
            wrapped = wrap_line(
                line.rstrip(), max_length=72, indent_level=indent_level
            )
            for wrapped_line in wrapped:
                result.append(wrapped_line + "\n")
        else:
            result.append(line)

        i += 1

    return result


def main():
    """Fix all docstrings in src directory."""
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    python_files.sort()

    fixed_count = 0

    for file in python_files:
        original_content = file.read_text()
        fixed_lines = fix_docstrings_in_file(file)
        fixed_content = "".join(fixed_lines)

        if original_content != fixed_content:
            file.write_text(fixed_content)
            print(f"Fixed: {file}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
