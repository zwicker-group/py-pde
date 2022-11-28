#!/usr/bin/env python3
"""
This script creates the requirements files in the project
"""

import csv
import subprocess as sp
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import List, Set

PACKAGE_PATH = Path(__file__).resolve().parents[1]


@dataclass
class Requirement:
    """simple class collecting data for a single required python package"""

    name: str  # name of the python package
    version: str  # minimal version
    usage: str = ""  # description for how the package is used in py-pde
    essential: bool = False  # basic requirement for the package
    docs_only: bool = False  # only required for creating documentation
    tests_only: bool = False  # only required for running tests
    collections: Set[str] = field(default_factory=set)  # collections where this fits

    @property
    def short_version(self) -> str:
        """str: simplified version string"""
        version = self.version
        processing = True
        while processing:
            if version.endswith(".0"):
                version = version[:-2]
            else:
                processing = False
        return version

    def line(self, relation: str = ">=") -> str:
        """create a line for a requirements file

        Args:
            relation (str): The relation used for version comparison

        Returns:
            str: A string that can be written to a requirements file
        """
        return f"{self.name}{relation}{self.version}"


REQUIREMENTS = [
    # essential requirements
    Requirement(
        name="matplotlib",
        version="3.1.0",
        usage="Visualizing results",
        essential=True,
    ),
    Requirement(
        name="numba",
        version="0.56.0",
        usage="Just-in-time compilation to accelerate numerics",
        essential=True,
    ),
    Requirement(
        name="numpy",
        version="1.22.0",
        usage="Handling numerical data",
        essential=True,
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
    Requirement(
        name="tqdm",
        version="4.60",
        usage="Display progress bars during calculations",
        essential=True,
    ),
    # general, optional requirements
    Requirement(
        name="h5py",
        version="2.10",
        usage="Storing data in the hierarchical file format",
        collections={"full", "multiprocessing", "docs"},
    ),
    Requirement(
        name="pandas",
        version="1.2",
        usage="Handling tabular data",
        collections={"full", "multiprocessing", "docs"},
    ),
    Requirement(
        name="pyfftw",
        version="0.12",
        usage="Faster Fourier transforms",
        collections={"full"},
    ),
    Requirement(
        name="ipywidgets",
        version="7",
        usage="Jupyter notebook support",
        collections={"interactive"},
    ),
    Requirement(
        name="mpi4py",
        version="3",
        usage="Parallel processing using MPI",
        collections={"multiprocessing"},
    ),
    Requirement(
        name="napari",
        version="0.4.8",
        usage="Displaying images interactively",
        collections={"interactive"},
    ),
    Requirement(
        name="numba-mpi",
        version="0.22",
        usage="Parallel processing using MPI+numba",
        collections={"multiprocessing"},
    ),
    # for documentation only
    Requirement(name="Sphinx", version="4", docs_only=True),
    Requirement(name="sphinx-autodoc-annotation", version="1.0", docs_only=True),
    Requirement(name="sphinx-gallery", version="0.6", docs_only=True),
    Requirement(name="sphinx-rtd-theme", version="0.4", docs_only=True),
    Requirement(name="Pillow", version="7.0", docs_only=True),
    # for tests only
    Requirement(name="jupyter_contrib_nbextensions", version="0.5", tests_only=True),
    Requirement(name="black", version="19.*", tests_only=True),
    Requirement(name="isort", version="5.1", tests_only=True),
    Requirement(name="mypy", version="0.770", tests_only=True),
    Requirement(name="pyinstrument", version="3", tests_only=True),
    Requirement(name="pytest", version="5.4", tests_only=True),
    Requirement(name="pytest-cov", version="2.8", tests_only=True),
    Requirement(name="pytest-xdist", version="1.30", tests_only=True),
]


SETUP_WARNING = (
    "# THIS FILE IS CREATED AUTOMATICALLY AND ALL MANUAL CHANGES WILL BE OVERWRITTEN\n"
    "# If you want to adjust settings in this file, change scripts/templates/{}\n\n"
)


def write_requirements_txt(
    path: Path,
    requirements: List[Requirement],
    *,
    relation: str = ">=",
    ref_base: bool = False,
    comment: str = None,
):
    """write requirements to a requirements.txt file

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
        relation (str): The relation that is used in the requirements file
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
            fp.write(reference.line(relation) + "\n")


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


def write_requirements_py(path: Path, requirements: List[Requirement]):
    """write requirements check into a python module

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
    """
    print(f"Modify `{path}`")

    # read user-created content of file
    content = []
    with open(path, "r") as fp:
        for line in fp:
            if "GENERATED CODE" in line:
                content.append(line)
                break
            content.append(line)
        else:
            raise ValueError("Could not find the token 'GENERATED CODE'")

    # add generated code
    for r in sorted(requirements, key=lambda r: r.name.lower()):
        content.append(f'check_package_version("{r.name}", "{r.version}")\n')
    content.append("del check_package_version\n")

    # write content back to file
    with open(path, "w") as fp:
        fp.writelines(content)


def write_from_template(
    path: Path,
    requirements: List[Requirement],
    template_name: str,
    fix_format: bool = False,
):
    """write file based on a template

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
        template_name (str): The name of the template
        fix_format (bool): If True, script will be formated using `black`
    """
    print(f"Write `{path}`")

    # format requirements list
    req_list = "[" + ", ".join('"' + ref.line(">=") + '"' for ref in requirements) + "]"

    # load template data
    template_path = Path(__file__).parent / "templates" / template_name
    with template_path.open("r") as fp:
        template = Template(fp.read())

    # format template
    substitutes = {"INSTALL_REQUIRES": req_list}
    for ref in REQUIREMENTS:
        substitutes[ref.name.replace("-", "_")] = ref.line(">=")
    content = template.substitute(substitutes)

    # write content to file
    with open(path, "w") as fp:
        fp.writelines(SETUP_WARNING.format(template_name) + content)

    # call black formatter on it
    if fix_format:
        sp.check_call(["black", "-q", "-t", "py38", str(path)])


def main():
    """main function creating all the requirements"""
    root = Path(PACKAGE_PATH)

    # write basic requirements
    write_requirements_txt(
        root / "requirements.txt",
        [r for r in REQUIREMENTS if r.essential],
    )

    # write minimal requirements to tests folder
    write_requirements_txt(
        root / "tests" / "requirements_min.txt",
        [r for r in REQUIREMENTS if r.essential],
        relation="~=",
        comment="These are the minimal requirements used to test compatibility",
    )

    # write full requirements to tests folder
    write_requirements_txt(
        root / "tests" / "requirements_full.txt",
        [r for r in REQUIREMENTS if r.essential or "full" in r.collections],
        comment="These are the full requirements used to test all functions",
    )

    # write full requirements to tests folder
    write_requirements_txt(
        root / "tests" / "requirements_mpi.txt",
        [r for r in REQUIREMENTS if r.essential or "multiprocessing" in r.collections],
        comment="These are requirements used to test multiprocessing",
    )

    # write requirements to tests folder
    write_requirements_txt(
        root / "tests" / "requirements.txt",
        [r for r in REQUIREMENTS if r.tests_only or "tests" in r.collections],
        ref_base=True,
    )

    # write requirements to docs folder
    write_requirements_txt(
        root / "docs" / "requirements.txt",
        [r for r in REQUIREMENTS if r.docs_only or "docs" in r.collections],
        ref_base=True,
    )

    # write requirements for documentation as CSV
    write_requirements_csv(
        root / "docs" / "source" / "_static" / "requirements_main.csv",
        [r for r in REQUIREMENTS if r.essential],
    )

    # write requirements for documentation as CSV
    write_requirements_csv(
        root / "docs" / "source" / "_static" / "requirements_optional.csv",
        [r for r in REQUIREMENTS if not (r.essential or r.tests_only or r.docs_only)],
    )

    # write setup.py
    write_from_template(
        root / "setup.py", [r for r in REQUIREMENTS if r.essential], "setup.py"
    )

    # write pyproject.toml
    write_from_template(
        root / "pyproject.toml",
        [r for r in REQUIREMENTS if r.essential],
        "pyproject.toml",
    )


if __name__ == "__main__":
    main()
