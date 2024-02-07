#!/usr/bin/env python3
"""
This script creates the requirements files in the project
"""

from __future__ import annotations

import csv
import subprocess as sp
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

PACKAGE_PATH = Path(__file__).resolve().parents[1]
MIN_PYTHON_VERSION = "3.9"
MAX_PYTHON_VERSION = "3.12"


@dataclass
class Requirement:
    """simple class collecting data for a single required python package"""

    name: str  # name of the python package
    version_min: str  # minimal version
    usage: str = ""  # description for how the package is used in py-pde
    relation: str | None = None  # relation used to compare version number
    essential: bool = False  # basic requirement for the package
    docs_only: bool = False  # only required for creating documentation
    tests_only: bool = False  # only required for running tests
    collections: set[str] = field(default_factory=set)  # collections where this fits

    @property
    def short_version(self) -> str:
        """str: simplified version_min string"""
        version = self.version_min.split(",")[0]  # only use the first part
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
            relation (str):
                The relation used for version_min comparison if self.relation is None

        Returns:
            str: A string that can be written to a requirements file
        """
        if self.relation is not None:
            relation = self.relation
        return f"{self.name}{relation}{self.version_min}"


REQUIREMENTS = [
    # essential requirements
    Requirement(
        name="matplotlib",
        version_min="3.1",
        usage="Visualizing results",
        essential=True,
    ),
    Requirement(
        name="numba",
        version_min="0.59",
        usage="Just-in-time compilation to accelerate numerics",
        essential=True,
    ),
    Requirement(
        name="numpy",
        version_min="1.22",
        usage="Handling numerical data",
        essential=True,
    ),
    Requirement(
        name="scipy",
        version_min="1.10",
        usage="Miscellaneous scientific functions",
        essential=True,
    ),
    Requirement(
        name="sympy",
        version_min="1.9",
        usage="Dealing with user-defined mathematical expressions",
        essential=True,
    ),
    Requirement(
        name="tqdm",
        version_min="4.66",
        usage="Display progress bars during calculations",
        essential=True,
    ),
    # general, optional requirements
    Requirement(
        name="h5py",
        version_min="2.10",
        usage="Storing data in the hierarchical file format",
        collections={"full", "multiprocessing", "docs"},
    ),
    Requirement(
        name="pandas",
        version_min="2",
        usage="Handling tabular data",
        collections={"full", "multiprocessing", "docs"},
    ),
    Requirement(
        name="pyfftw",
        version_min="0.12",
        usage="Faster Fourier transforms",
        collections={},  # include in "full" collection when pyfftw supports python 3.12
    ),
    Requirement(
        name="rocket-fft",
        version_min="0.2.4",
        usage="Numba-compiled fast Fourier transforms",
        collections={"full"},
    ),
    Requirement(
        name="ipywidgets",
        version_min="8",
        usage="Jupyter notebook support",
        collections={"interactive"},
    ),
    Requirement(
        name="mpi4py",
        version_min="3",
        usage="Parallel processing using MPI",
        collections={"multiprocessing"},
    ),
    Requirement(
        name="napari",
        version_min="0.4.8",
        usage="Displaying images interactively",
        collections={"interactive"},
    ),
    Requirement(
        name="numba-mpi",
        version_min="0.22",
        usage="Parallel processing using MPI+numba",
        collections={"multiprocessing"},
    ),
    # for documentation only
    Requirement(name="Sphinx", version_min="4", docs_only=True),
    Requirement(name="sphinx-autodoc-annotation", version_min="1.0", docs_only=True),
    Requirement(name="sphinx-gallery", version_min="0.6", docs_only=True),
    Requirement(name="sphinx-rtd-theme", version_min="1", docs_only=True),
    Requirement(name="Pillow", version_min="7.0", docs_only=True),
    # for tests only
    Requirement(
        name="jupyter_contrib_nbextensions", version_min="0.5", tests_only=True
    ),
    Requirement(name="black", version_min="24.*", tests_only=True),
    Requirement(name="importlib-metadata", version_min="5", tests_only=True),
    Requirement(name="isort", version_min="5.1", tests_only=True),
    Requirement(name="mypy", version_min="1.8", tests_only=True),
    Requirement(name="notebook", version_min="7", tests_only=True),
    Requirement(name="pyupgrade", version_min="3", tests_only=True),
    Requirement(name="pytest", version_min="5.4", tests_only=True),
    Requirement(name="pytest-cov", version_min="2.8", tests_only=True),
    Requirement(name="pytest-xdist", version_min="1.30", tests_only=True),
]


SETUP_WARNING = (
    "# THIS FILE IS CREATED AUTOMATICALLY AND ALL MANUAL CHANGES WILL BE OVERWRITTEN\n"
    "# If you want to adjust settings in this file, change scripts/_templates/{}\n\n"
)


def write_requirements_txt(
    path: Path,
    requirements: list[Requirement],
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
    path.parent.mkdir(exist_ok=True, parents=True)  # ensure path exists
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
    path: Path, requirements: list[Requirement], *, incl_version: bool = True
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


def write_requirements_py(path: Path, requirements: list[Requirement]):
    """write requirements check into a python module

    Args:
        path (:class:`Path`): The path where the requirements are written
        requirements (list): The requirements to be written
    """
    print(f"Modify `{path}`")

    # read user-created content of file
    content = []
    with open(path) as fp:
        for line in fp:
            if "GENERATED CODE" in line:
                content.append(line)
                break
            content.append(line)
        else:
            raise ValueError("Could not find the token 'GENERATED CODE'")

    # add generated code
    for r in sorted(requirements, key=lambda r: r.name.lower()):
        content.append(f'check_package_version("{r.name}", "{r.version_min}")\n')
    content.append("del check_package_version\n")

    # write content back to file
    with open(path, "w") as fp:
        fp.writelines(content)


def write_from_template(
    path: Path,
    template_name: str,
    *,
    requirements: list[Requirement] | None = None,
    fix_format: bool = False,
    add_warning: bool = True,
):
    """write file based on a template

    Args:
        path (:class:`Path`): The path where the requirements are written
        template_name (str): The name of the template
        requirements (list): The requirements to be written
        fix_format (bool): If True, script will be formated using `black`
        add_warning (bool): If True, adds a warning that file is generated
    """
    print(f"Write `{path}`")

    # load template data
    template_path = Path(__file__).parent / "_templates" / template_name
    with template_path.open("r") as fp:
        template = Template(fp.read())

    # parse python version_min
    major, minor = MAX_PYTHON_VERSION.split(".")
    minor_next = int(minor) + 1

    # determine template substitutes
    substitutes = {
        "MIN_PYTHON_VERSION": MIN_PYTHON_VERSION,
        "MIN_PYTHON_VERSION_NODOT": MIN_PYTHON_VERSION.replace(".", ""),
        "MAX_PYTHON_VERSION": MAX_PYTHON_VERSION,
        "MAX_PYTHON_VERSION_NEXT": f"{major}.{minor_next}",
    }
    if requirements:
        req_list = (
            "[" + ", ".join('"' + ref.line(">=") + '"' for ref in requirements) + "]"
        )
        substitutes["INSTALL_REQUIRES"] = req_list
    for ref in REQUIREMENTS:
        substitutes[ref.name.replace("-", "_")] = ref.line(">=")
    content = template.substitute(substitutes)

    # write content to file
    with open(path, "w") as fp:
        if add_warning:
            fp.writelines(SETUP_WARNING.format(template_name))
        fp.writelines(content)

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
    # write basic requirements
    write_requirements_txt(
        root / "pde" / "tools" / "resources" / "requirements_basic.txt",
        [r for r in REQUIREMENTS if r.essential],
        comment="These are the basic requirements for the package",
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
        root / "pde" / "tools" / "resources" / "requirements_full.txt",
        [r for r in REQUIREMENTS if r.essential or "full" in r.collections],
        comment="These are the full requirements used to test all functions",
    )

    # write full requirements to tests folder
    write_requirements_txt(
        root / "tests" / "requirements_mpi.txt",
        [r for r in REQUIREMENTS if r.essential or "multiprocessing" in r.collections],
        comment="These are requirements used to test multiprocessing",
    )
    write_requirements_txt(
        root / "pde" / "tools" / "resources" / "requirements_mpi.txt",
        [r for r in REQUIREMENTS if r.essential or "multiprocessing" in r.collections],
        comment="These are requirements for supporting multiprocessing",
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

    # write pyproject.toml
    write_from_template(
        root / "pyproject.toml",
        "pyproject.toml",
        requirements=[r for r in REQUIREMENTS if r.essential],
    )

    # write pyproject.toml
    write_from_template(root / "runtime.txt", "runtime.txt", add_warning=False)


if __name__ == "__main__":
    main()
