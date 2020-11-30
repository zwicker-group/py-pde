#!/usr/bin/env python3

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path

PACKAGE = "pde"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package


def test_codestyle(*, verbose: bool = True):
    """run the codestyle tests

    Args:
        verbose (bool): Whether to do extra output
    """
    for folder in [PACKAGE, "examples"]:
        if verbose:
            print(f"Checking codestyle in folder {folder}...")
        path = PACKAGE_PATH / folder

        # format imports
        sp.run(["isort", "--profile", "black", "--diff", path])
        # format rest
        sp.run(["black", "-t", "py36", "--check", path])


def test_types(*, report: bool = False, verbose: bool = True):
    """run mypy to check the types of the python code

    Args:
        report (bool): Whether to write a report
        verbose (bool): Whether to do extra output
    """
    if verbose:
        print(f"Checking types in the {PACKAGE} package...")

    args = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        PACKAGE_PATH / "tests" / "mypy.ini",
    ]

    if report:
        folder = PACKAGE_PATH / "tests" / "mypy-report"
        if verbose:
            print(f"Writing report to `{folder}`")
        args.extend(
            ["--no-incremental", "--linecount-report", folder, "--html-report", folder]
        )
    else:
        # do not create a report
        args.append("--pretty")

    args.extend(["--package", PACKAGE])

    sp.run(args, cwd=PACKAGE_PATH)


def run_unit_tests(
    runslow: bool = False,
    parallel: bool = False,
    coverage: bool = False,
    no_numba: bool = False,
    pattern: str = None,
):
    """run the unit tests

    Args:
        runslow (bool): Whether to run the slow tests
        parallel (bool): Whether to use multiple processors
        coverage (bool): Whether to determine the test coverage
        no_numba (bool): Whether to disable numba jit compilation
        pattern (str): A pattern that determines which tests are ran
    """
    # modify current environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "agg"
    if no_numba:
        env["NUMBA_DISABLE_JIT"] = "1"
    else:
        env["NUMBA_WARNINGS"] = "1"
        env["NUMBA_BOUNDSCHECK"] = "1"

    # build the arguments string
    args = [
        sys.executable,
        "-m",
        "pytest",  # run pytest module
        "-c",
        "tests/pytest.ini",  # locate the configuration file
        "-rs",  # show summary of skipped tests
        "-rw",  # show summary of warnings raised during tests
    ]

    # allow running slow tests?
    if runslow:
        args.append("--runslow")

    # run tests using multiple cores?
    if parallel:
        from multiprocessing import cpu_count

        args.extend(["-n", str(cpu_count() // 2), "--durations=10"])

    # run only a subset of the tests?
    if pattern is not None:
        args.extend(["-k", str(pattern)])

    # add coverage attributes?
    if coverage:
        args.extend(
            [
                "--cov-config=tests/.coveragerc",  # locate the configuration file
                "--cov-report",
                "html:tests/coverage",  # create a report in html format
                f"--cov={PACKAGE}",  # specify in which package the coverage is measured
            ]
        )

    # specify the package to run
    args.append(PACKAGE)

    # actually run the test
    sp.run(args, env=env, cwd=PACKAGE_PATH)


def main():
    """ the main program controlling the tests """
    # parse the command line arguments
    parser = argparse.ArgumentParser(
        description=f"Run tests of the `{PACKAGE}` package.",
        epilog="All test categories are run if no specific categories are selected.",
    )

    # add the basic tests that need to be run
    group = parser.add_argument_group("Test categories")
    group.add_argument(
        "-s", "--style", action="store_true", default=False, help="Test code style"
    )
    group.add_argument(
        "-t", "--types", action="store_true", default=False, help="Test object types"
    )
    group.add_argument(
        "-u", "--unit", action="store_true", default=False, help="Run unit tests"
    )

    # set additional arguments
    group = parser.add_argument_group("Additional arguments")
    group.add_argument(
        "-q",
        "--quite",
        action="store_true",
        default=False,
        help="Suppress output from the script",
    )
    group.add_argument(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run slow unit tests",
    )
    group.add_argument(
        "--coverage",
        action="store_true",
        default=False,
        help="Record test coverage of unit tests",
    )
    group.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use multiprocessing",
    )
    group.add_argument(
        "--no_numba",
        action="store_true",
        default=False,
        help="Do not use just-in-time compilation of numba",
    )
    group.add_argument(
        "--pattern",
        metavar="PATTERN",
        type=str,
        help="Only run tests with this pattern",
    )
    group.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Write a report of the results",
    )

    # parse the command line arguments
    args = parser.parse_args()
    run_all = not (args.style or args.types or args.unit)

    # run the requested tests
    if run_all or args.style:
        test_codestyle(verbose=not args.quite)
    if run_all or args.types:
        test_types(report=args.report, verbose=not args.quite)
    if run_all or args.unit:
        run_unit_tests(
            runslow=args.runslow,
            coverage=args.coverage,
            parallel=args.parallel,
            no_numba=args.no_numba,
            pattern=args.pattern,
        )


if __name__ == "__main__":
    main()
