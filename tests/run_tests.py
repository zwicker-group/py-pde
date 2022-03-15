#!/usr/bin/env python3

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import Sequence

PACKAGE = "pde"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package


def _most_severe_exit_code(retcodes: Sequence[int]) -> int:
    """returns the most severe exit code of a given list

    Args:
        retcodes (list): A list of return codes

    Returns:
        int: the exit code that is most severe
    """
    if all(retcode == 0 for retcode in retcodes):
        return 0
    else:
        return max(retcodes, key=lambda retcode: abs(retcode))


def show_config():
    """show package configuration"""
    from importlib.machinery import SourceFileLoader

    # imports the package from the package path
    pde = SourceFileLoader(PACKAGE, f"{PACKAGE_PATH/PACKAGE}/__init__.py").load_module()
    # obtain the package environment
    env = pde.environment()

    print(f"{'='*33} CONFIGURATION {'='*32}")
    for category, data in env.items():
        if hasattr(data, "items"):
            print(f"\n{category}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            data_formatted = data.replace("\n", "\n    ")
            print(f"{category}: {data_formatted}")
    print("=" * 80)


def test_codestyle(*, verbose: bool = True) -> int:
    """run the codestyle tests

    Args:
        verbose (bool): Whether to do extra output

    Returns:
        int: The most severe exit code
    """
    retcodes = []

    for folder in [PACKAGE, "examples"]:
        if verbose:
            print(f"Checking codestyle in folder {folder}...")
        path = PACKAGE_PATH / folder

        # format imports
        result = sp.run(["isort", "--profile", "black", "--diff", path])
        retcodes.append(result.returncode)
        # format rest
        result = sp.run(["black", "-t", "py36", "--check", path])
        retcodes.append(result.returncode)

    return _most_severe_exit_code(retcodes)


def test_types(*, report: bool = False, verbose: bool = True) -> int:
    """run mypy to check the types of the python code

    Args:
        report (bool): Whether to write a report
        verbose (bool): Whether to do extra output

    Returns:
        int: The return code indicating success or failure
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

    return sp.run(args, cwd=PACKAGE_PATH).returncode


def run_unit_tests(
    runslow: bool = False,
    runinteractive: bool = False,
    parallel: bool = False,
    coverage: bool = False,
    nojit: bool = False,
    early: bool = False,
    pattern: str = None,
) -> int:
    """run the unit tests

    Args:
        runslow (bool): Whether to run the slow tests
        runinteractive (bool): Whether to run the interactive tests
        parallel (bool): Whether to use multiple processors
        coverage (bool): Whether to determine the test coverage
        nojit (bool): Whether to disable numba jit compilation
        early (bool): Whether to fail at the first test
        pattern (str): A pattern that determines which tests are ran

    Returns:
        int: The return code indicating success or failure
    """
    # modify current environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "agg"
    if nojit:
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
        "--import-mode=importlib",
    ]

    # allow running slow and interactive tests?
    if runslow:
        args.append("--runslow")
    if runinteractive:
        args.append("--runinteractive")

    # fail early if requested
    if early:
        args.append("--maxfail=1")

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
    return sp.run(args, env=env, cwd=PACKAGE_PATH).returncode


def main() -> int:
    """the main program controlling the tests

    Returns:
        int: The return code indicating success or failure
    """
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
        "--runinteractive",
        action="store_true",
        default=False,
        help="Also run interactive unit tests",
    )
    group.add_argument(
        "--showconfig",
        action="store_true",
        default=False,
        help="Show configuration before running tests",
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
        "--nojit",
        action="store_true",
        default=False,
        help="Do not use just-in-time compilation of numba",
    )
    group.add_argument(
        "--early",
        action="store_true",
        default=False,
        help="Return at first failed test",
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
    run_all = not (args.style or args.types or args.unit or args.showconfig)

    # show the package configuration
    if args.showconfig:
        show_config()

    # run the requested tests
    retcodes = []
    if run_all or args.style:
        retcode = test_codestyle(verbose=not args.quite)
        retcodes.append(retcode)

    if run_all or args.types:
        retcode = test_types(report=args.report, verbose=not args.quite)
        retcodes.append(retcode)

    if run_all or args.unit:
        retcode = run_unit_tests(
            runslow=args.runslow,
            runinteractive=args.runinteractive,
            coverage=args.coverage,
            parallel=args.parallel,
            nojit=args.nojit,
            early=args.early,
            pattern=args.pattern,
        )
        retcodes.append(retcode)

    # return the most severe code
    return _most_severe_exit_code(retcodes)


if __name__ == "__main__":
    sys.exit(main())
