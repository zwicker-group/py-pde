#!/usr/bin/env python3

import argparse
import os
import subprocess as sp
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]


def test_codestyle():
    """ run the codestyle tests """
    for folder in ["pde", "examples"]:
        print(f"Checking codestyle in folder {folder}:")
        path = PACKAGE_PATH / folder

        # format imports
        sp.check_call(["isort", "--profile", "black", "--diff", path])
        sp.check_call(["black", "-t", "py36", "--check", path])


def test_types():
    """ run mypy to check the types of the python code """
    sp.check_call(
        [
            "python3",
            "-m",
            "mypy",
            "--config-file",
            str(PACKAGE_PATH / "tests" / "mypy.ini"),
            "--pretty",
            "--package",
            "pde",
        ]
    )


def run_tests(runslow: bool = False, parallel: bool = False, pattern: str = None):
    """ run the actual tests
    
    Args:
        runslow (bool): Whether to run the slow tests
        parallel (bool): Whether to use multiple processors
        pattern (str): A pattern that determines which tests are ran
    """
    # modify current environment
    env = os.environ.copy()
    env["NUMBA_WARNINGS"] = "1"
    env["NUMBA_BOUNDSCHECK"] = "1"
    env["MPLBACKEND"] = "agg"

    # build the arguments string
    args = ["python3", "-m", "pytest", "-c", "tests/pytest.ini", "-rs"]

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

    # specify the package to run
    args.extend(["pde"])

    # actually run the test
    sp.check_call(args, env=env, cwd=PACKAGE_PATH)


def main():
    """ the main program controlling the tests """
    # parse the command line arguments
    parser = argparse.ArgumentParser(description="Run tests")

    # add the basic tests that need to be run
    parser.add_argument(
        "-r", "--run", action="store_true", default=False, help="Run tests"
    )
    parser.add_argument(
        "-t", "--types", action="store_true", default=False, help="Test object types"
    )
    parser.add_argument(
        "-s", "--style", action="store_true", default=False, help="Test code style"
    )

    # set additional arguments
    parser.add_argument(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run slow tests",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use multiprocessing",
    )
    parser.add_argument(
        "--pattern",
        metavar="PATTERN",
        type=str,
        help="Only run tests with this pattern",
    )

    # parse the command line arguments
    args = parser.parse_args()
    run_all = not (args.style or args.types or args.run)

    # run the requested tests
    if run_all or args.style:
        test_codestyle()
    if run_all or args.types:
        test_types()
    if run_all or args.run:
        run_tests(runslow=args.runslow, parallel=args.parallel, pattern=args.pattern)


if __name__ == "__main__":
    main()
