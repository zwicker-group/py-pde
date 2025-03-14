#!/usr/bin/env python3

import glob
import logging
import os
import subprocess as sp
from pathlib import Path

logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = "packages"
REPLACEMENTS = {
    "Submodules\n----------\n\n": "",
    "Subpackages\n-----------": "**Subpackages:**",
    "pde package\n===========": "Reference manual\n================",
}


def replace_in_file(infile, replacements, outfile=None):
    """Reads in a file, replaces the given data using python formatting and writes back
    the result to a file.

    Args:
        infile (str):
            File to be read
        replacements (dict):
            The replacements old => new in a dictionary format {old: new}
        outfile (str):
            Output file to which the data is written. If it is omitted, the
            input file will be overwritten instead
    """
    if outfile is None:
        outfile = infile

    with Path(infile).open() as fp:
        content = fp.read()

    for key, value in replacements.items():
        content = content.replace(key, value)

    with Path(outfile).open("w") as fp:
        fp.write(content)


def main():
    """Run the autodoc call."""
    logger = logging.getLogger("autodoc")

    # remove old files
    for path in Path(OUTPUT_PATH).glob("*.rst"):
        logger.info("Remove file `%s`", path)
        path.unlink()

    # run sphinx-apidoc
    sp.check_call(
        [
            "sphinx-apidoc",
            "--separate",
            "--maxdepth",
            "4",
            "--output-dir",
            OUTPUT_PATH,
            "--module-first",
            "../../pde",  # path of the package
            "../../pde/version.py",  # ignored file
            "../../**/conftest.py",  # ignored file
            "../../**/tests",  # ignored path
        ]
    )

    # replace unwanted information
    for path in Path(OUTPUT_PATH).glob("*.rst"):
        logger.info("Patch file `%s`", path)
        replace_in_file(path, REPLACEMENTS)


if __name__ == "__main__":
    main()
