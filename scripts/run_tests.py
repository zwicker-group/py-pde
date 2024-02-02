#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from collections.abc import Sequence
from pathlib import Path

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
    mod = SourceFileLoader(PACKAGE, f"{PACKAGE_PATH/PACKAGE}/__init__.py").load_module()
    # obtain the package environment
    env = mod.environment()

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
        result = sp.run(["isort", "--diff", path])
        retcodes.append(result.returncode)
        # format rest
        result = sp.run(["black", "--check", path])
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
        PACKAGE_PATH / "pyproject.toml",
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
    *,
    runslow: bool = False,
    runinteractive: bool = False,
    use_mpi: bool = False,
    num_cores: str | int = 1,
    coverage: bool = False,
    nojit: bool = False,
    early: bool = False,
    pattern: str = None,
) -> int:
    """run the unit tests

    Args:
        runslow (bool): Whether to run the slow tests
        runinteractive (bool): Whether to run the interactive tests
        use_mpi (bool): Flag indicating whether tests are run using MPI
        num_cores (int or str): Number of cores to use (`auto` for automatic choice)
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

    if use_mpi:
        # run pytest using MPI with two cores
        args = ["mpiexec", "-n", "2"]
        if sys.platform == "darwin":
            args += ["-host", "localhost:2"]
    else:
        args = []

    # build the arguments string
    args += [
        sys.executable,
        "-m",
        "pytest",  # run pytest module
        "-c",
        "pyproject.toml",  # locate the configuration file
        "-rs",  # show summary of skipped tests
        "-rw",  # show summary of warnings raised during tests
        # "--import-mode=importlib",
    ]

    if runslow:
        args.append("--runslow")  # also run slow tests
    if runinteractive:
        args.append("--runinteractive")  # also run interactive tests
    if use_mpi:
        try:
            import numba_mpi  # @UnusedImport
        except ImportError:
            raise RuntimeError("Moduled `numba_mpi` is required to test with MPI")
        args.append("--use_mpi")  # only run tests requiring MPI multiprocessing

    # fail early if requested
    if early:
        args.append("--maxfail=1")

    # run tests using multiple cores?
    if num_cores == "auto":
        num_cores = os.cpu_count()
    else:
        num_cores = int(num_cores)
    if num_cores > 1:
        args.extend(["-n", str(num_cores), "--durations=10"])

    # run only a subset of the tests?
    if pattern is not None:
        args.extend(["-k", str(pattern)])

    # add coverage attributes?
    if coverage:
        env["PYPDE_TESTRUN"] = "1"
        args.extend(
            [
                "--cov-config=pyproject.toml",  # locate the configuration file
                "--cov-report",
                "html:scripts/coverage",  # create a report in html format
                f"--cov={PACKAGE}",  # specify in which package the coverage is measured
            ]
        )
        if use_mpi:
            # this is a hack to allow appending the coverage report
            args.append("--cov-append")

    # specify the package to run
    args.append("tests")

    # actually run the test
    retcode = sp.run(args, env=env, cwd=PACKAGE_PATH).returncode

    # delete intermediate coverage files, which are sometimes left behind
    if coverage:
        for p in Path("..").glob(".coverage*"):
            p.unlink()

    return retcode


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
        "--use_mpi",
        action="store_true",
        default=False,
        help="Run each unit test with MPI multiprocessing",
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
        "--num_cores",
        metavar="CORES",
        type=str,
        default=1,
        help="Number of cores to use (`auto` for automatic choice)",
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
            use_mpi=args.use_mpi,
            coverage=args.coverage,
            num_cores=args.num_cores,
            nojit=args.nojit,
            early=args.early,
            pattern=args.pattern,
        )
        retcodes.append(retcode)

    # return the most severe code
    return _most_severe_exit_code(retcodes)


if __name__ == "__main__":
    sys.exit(main())
