#!/usr/bin/env python3
"""
This script creates the requirements files in the project
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

PACKAGE_PATH = Path(__file__).resolve().parents[1]


@dataclass
class Requirement:
    """ simple class collecting data for a single required python package """

    name: str  # name of the python package
    version: str  # minimal version
    usage: str = ""  # description for how the package is used in py-pde
    essential: bool = False  # basic requirement for the package
    optional: bool = False  # optional requirements for the package
    for_docs: bool = False  # required for documentation
    for_tests: bool = False  # required for tests

    @property
    def short_version(self):
        version = self.version
        processing = True
        while processing:
            if version.endswith(".0"):
                version = version[:-2]
            else:
                processing = False
        return version

    def line(self, rel: str = ">=") -> str:
        return f"{self.name}{rel}{self.version}"


REQUIREMENTS = [
    # essential requirements
    Requirement(
        name="matplotlib", version="3.1.0", usage="Visualizing results", essential=True
    ),
    Requirement(
        name="numba",
        version="0.50.0",
        usage="Just-in-time compilation to accelerate numerics",
        essential=True,
    ),
    Requirement(
        name="numpy", version="1.18.0", usage="Handling numerical data", essential=True
    ),
    Requirement(
        name="scipy",
        version="1.4.0",
        usage="Miscellaneous scientific functions",
        essential=True,
    ),
    Requirement(
        name="sympy",
        version="1.5.0",
        usage="Dealing with user-defined mathematical expressions",
        essential=True,
    ),
    # general, optional requirements
    Requirement(
        name="h5py",
        version="2.10",
        usage="Storing data in the hierarchical file format",
        optional=True,
        for_docs=True,
        for_tests=True,
    ),
    Requirement(
        name="napari",
        version="0.4",
        usage="Displaying images interactively",
        optional=True,
    ),
    Requirement(
        name="pandas",
        version="1.2",
        usage="Handling tabular data",
        optional=True,
        for_docs=True,
        for_tests=True,
    ),
    Requirement(
        name="pyfftw",
        version="0.12",
        usage="Faster Fourier transforms",
        optional=True,
    ),
    Requirement(
        name="tqdm",
        version="4.45",
        usage="Display progress bars during calculations",
        optional=True,
        for_docs=True,
        for_tests=True,
    ),
    # for documentation only
    Requirement(name="sphinx-autodoc-annotation", version="1.0", for_docs=True),
    Requirement(name="sphinx-gallery", version="0.6", for_docs=True),
    Requirement(name="sphinx-rtd-theme", version="0.4", for_docs=True),
    Requirement(name="Pillow", version="7.0", for_docs=True),
    # for tests only
    Requirement(name="black", version="19.*", for_tests=True),
    Requirement(name="isort", version="5.1", for_tests=True),
    Requirement(name="mypy", version="0.770", for_tests=True),
    Requirement(name="pyinstrument", version="3", for_tests=True),
    Requirement(name="pytest", version="5.4", for_tests=True),
    Requirement(name="pytest-cov", version="2.8", for_tests=True),
    Requirement(name="pytest-xdist", version="1.30", for_tests=True),
]


def write_requirements_txt(
    path: Path,
    requirements: List[Requirement],
    *,
    rel: str = ">=",
    ref_base: bool = False,
    comment: str = None,
):
    """write requirements to a requirements.txt file

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
        rel (str): The relation that is used in the requirements file
        ref_base (bool): Whether the basic requirements.txt is referenced
        comment (str): An optional comment on top of the requirements file
    """
    print(f"Write `{path}`")
    with open(path, "w") as fp:
        if comment:
            fp.write(f"# {comment}\n")
        if ref_base:
            levels = len(path.parent.relative_to(PACKAGE_PATH).parts)
            reference = Path("/".join([".."] * levels)) / "requirements.txt"
            fp.write(f"-r {reference}\n")
        for reference in sorted(requirements, key=lambda r: r.name.lower()):
            fp.write(reference.line(rel) + "\n")


def write_requirements_csv(
    path: Path, requirements: List[Requirement], *, incl_version: bool = True
):
    """write requirements to a CSV file

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
    """
    print(f"Write `{path}`")
    with open(path, "w") as fp:
        writer = csv.writer(fp)
        if incl_version:
            writer.writerow(["Package", "Minimal version", "Usage"])
        else:
            writer.writerow(["Package", "Usage"])
        for r in sorted(requirements, key=lambda r: r.name.lower()):
            if incl_version:
                writer.writerow([r.name, r.short_version, r.usage])
            else:
                writer.writerow([r.name, r.usage])


def main():
    """ main function creating all the requirements """
    # write basic requirements
    write_requirements_txt(
        Path(PACKAGE_PATH) / "requirements.txt",
        [r for r in REQUIREMENTS if r.essential],
    )

    # write minimal requirements to tests folder
    write_requirements_txt(
        Path(PACKAGE_PATH) / "tests" / "requirements_min.txt",
        [r for r in REQUIREMENTS if r.essential],
        rel="~=",
        comment="These are the minimal requirements used to test compatibility",
    )

    # write requirements to docs folder
    write_requirements_txt(
        Path(PACKAGE_PATH) / "docs" / "requirements.txt",
        [r for r in REQUIREMENTS if r.for_docs],
        ref_base=True,
    )

    # write requirements to docs folder
    write_requirements_txt(
        Path(PACKAGE_PATH) / "tests" / "requirements.txt",
        [r for r in REQUIREMENTS if r.for_tests],
        ref_base=True,
    )

    # write requirements for documentation as CSV
    write_requirements_csv(
        Path(PACKAGE_PATH) / "docs" / "source" / "_static" / "requirements_main.csv",
        [r for r in REQUIREMENTS if r.essential],
    )

    # write requirements for documentation as CSV
    write_requirements_csv(
        Path(PACKAGE_PATH)
        / "docs"
        / "source"
        / "_static"
        / "requirements_optional.csv",
        [r for r in REQUIREMENTS if r.optional],
        incl_version=False,
    )


if __name__ == "__main__":
    main()
