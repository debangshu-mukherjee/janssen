#!/usr/bin/env python3
"""
Automatically update the tutorials index.rst file to include all Jupyter notebooks.
Run this script whenever new notebooks are added to the tutorials directory.
"""

import os
from pathlib import Path


def update_tutorials_index():
    """Update the tutorials/index.rst file with all notebooks found."""

    # Get paths
    docs_dir = Path(__file__).parent
    tutorials_dir = docs_dir.parent / "tutorials"
    index_file = tutorials_dir / "index.rst"

    # Find all notebook files
    notebooks = sorted([f.stem for f in tutorials_dir.glob("*.ipynb")])

    if not notebooks:
        print("No notebooks found in tutorials directory")
        return

    # Create the index content
    content = """Tutorials
=========

This section contains interactive Jupyter notebooks demonstrating how to use Janssen for various applications.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Notebooks

"""

    # Add each notebook
    for notebook in notebooks:
        content += f"   {notebook}\n"

    content += """
.. note::

   These notebooks are rendered automatically from the ``tutorials/`` directory.
   To run them interactively:

   1. Clone the repository
   2. Navigate to the ``tutorials/`` directory
   3. Launch Jupyter: ``jupyter notebook`` or ``jupyter lab``
   4. Open any notebook to explore the examples
"""

    # Write the file
    with open(index_file, "w") as f:
        f.write(content)

    print(f"Updated {index_file} with {len(notebooks)} notebooks:")
    for notebook in notebooks:
        print(f"  - {notebook}")


if __name__ == "__main__":
    update_tutorials_index()
