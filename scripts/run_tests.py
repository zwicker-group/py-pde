#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

PACKAGE = "pde"  # name of the package that needs to be tested
PACKAGE_PATH = Path(__file__).resolve().parents[1]  # base path of the package


def _most_severe_exit_code(retcodes: Sequence[int]) -> int:
    """Returns the most severe exit code of a given list.

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
    """Show package configuration."""
    from importlib.machinery import SourceFileLoader

    # imports the package from the package path
    mod_path = f"{PACKAGE_PATH / PACKAGE}/__init__.py"
    mod = SourceFileLoader(PACKAGE, mod_path).load_module()
    # obtain the package environment
    env = mod.environment()

    print(f"{'=' * 33} CONFIGURATION {'=' * 32}")
    for category, data in env.items():
        if hasattr(data, "items"):
            print(f"\n{category}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            data_formatted = data.replace("\n", "\n    ")
            print(f"{category}: {data_formatted}")
    print("=" * 80)


def run_test_codestyle(*, verbose: bool = True) -> int:
    """Run the codestyle tests.

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

        # check format
        result = sp.run(["ruff", "check", path])
        retcodes.append(result.returncode)

    return _most_severe_exit_code(retcodes)


def run_test_types(*, report: bool = False, verbose: bool = True) -> int:
    """Run mypy to check the types of the python code.

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
    pattern: str = None,
    use_memray: bool = False,
    pytest_args: list[str] | None = None,
) -> int:
    """Run the unit tests.

    Args:
        runslow (bool):
            Whether to run the slow tests
        runinteractive (bool):
            Whether to run the interactive tests
        use_mpi (bool):
            Flag indicating whether tests are run using MPI
        num_cores (int or str):
            Number of cores to use (`auto` for automatic choice)
        coverage (bool):
            Whether to determine the test coverage
        nojit (bool):
            Whether to disable numba jit compilation
        pattern (str):
            A pattern that determines which tests are ran
        use_memray (bool):
            Use memray to trace memory allocations during tests
        pytest_args (list of str):
            Additional arguments forwarded to py.test. For instance ["--maxfail=1"]
            fails tests early.

    Returns:
        int: The return code indicating success or failure
    """
    if pytest_args is None:
        pytest_args = []

    # modify current environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "agg"
    if nojit:
        env["NUMBA_DISABLE_JIT"] = "1"
    else:
        env["NUMBA_WARNINGS"] = "1"
        env["NUMBA_BOUNDSCHECK"] = "1"

    # determine how to invoke the python interpreter
    if use_mpi:
        # run pytest using MPI with two cores
        args = ["mpiexec", "-n", "2"]
        if sys.platform == "darwin":
            args += ["-host", "localhost:2"]
    else:
        args = []
    args += [sys.executable]

    # add the arguments to invoke memory tracing
    if use_memray:
        memray_file = Path(tempfile.gettempdir()) / f"memray-{PACKAGE}.bin"
        memray_file.unlink(missing_ok=True)
        args += ["-m", "memray", "run", "--aggregate", "--output", str(memray_file)]

    # add the arguments to invoke the testing
    args += [
        "-m",
        "pytest",  # run pytest module
        "-c",
        "pyproject.toml",  # locate the configuration file
        "-rs",  # show summary of skipped tests
        "-rw",  # show summary of warnings raised during tests
    ]

    # add extra arguments for special testing situations
    if runslow:
        args.append("--runslow")  # also run slow tests
    if runinteractive:
        args.append("--runinteractive")  # also run interactive tests
    if use_mpi:
        try:
            import numba_mpi
        except ImportError as err:
            raise RuntimeError(
                "Module `numba_mpi` is required to test with MPI"
            ) from err
        args.append("--use_mpi")  # only run tests requiring MPI multiprocessing

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
        args += [
            "--cov-config=pyproject.toml",  # locate the configuration file
            "--cov-report",
            "html:scripts/coverage",  # create a report in html format
            f"--cov={PACKAGE}",  # specify in which package the coverage is measured
        ]

    args.extend(pytest_args)

    # specify the package to run
    args += ["tests"]

    # actually run the test
    retcode = sp.run(args, env=env, cwd=PACKAGE_PATH).returncode

    # delete intermediate coverage files, which are sometimes left behind
    if coverage:
        for p in Path("..").glob(".coverage*"):
            p.unlink()

    # post-process memory traces
    if use_memray:
        # show a summary of the memory usage
        args = [sys.executable, "-m", "memray", "summary", str(memray_file)]
        retcode1 = sp.run(args, env=env).returncode
        # create a flamegraph of the memory usage
        flamegraph_file = Path("flamegraph.html")
        flamegraph_file.unlink(missing_ok=True)
        args = [
            sys.executable,
            "-m",
            "memray",  # use memray
            "flamegraph",  # create a flamegraph to summarize output
            str(memray_file),
            "-o",
            str(flamegraph_file),
        ]
        retcode2 = sp.run(args, env=env).returncode
        retcode = _most_severe_exit_code([retcode, retcode1, retcode2])

    return retcode


def main() -> int:
    """The main program controlling the tests.

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
        "--use_memray",
        action="store_true",
        default=False,
        help="Run test suite under memray to trace memory allocations",
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

    # set py.test arguments
    group = parser.add_argument_group(
        "py.test arguments",
        description="Additional arguments separated by `--` are forward to py.test",
    )
    group.add_argument("pytest_args", nargs="*", help=argparse.SUPPRESS)

    # parse the command line arguments
    args = parser.parse_args()
    run_all = not (args.style or args.types or args.unit or args.showconfig)

    # show the package configuration
    if args.showconfig:
        show_config()

    # run the requested tests
    retcodes = []
    if run_all or args.style:
        retcode = run_test_codestyle(verbose=not args.quite)
        retcodes.append(retcode)

    if run_all or args.types:
        retcode = run_test_types(report=args.report, verbose=not args.quite)
        retcodes.append(retcode)

    if run_all or args.unit:
        retcode = run_unit_tests(
            runslow=args.runslow,
            runinteractive=args.runinteractive,
            use_mpi=args.use_mpi,
            use_memray=args.use_memray,
            coverage=args.coverage,
            num_cores=args.num_cores,
            nojit=args.nojit,
            pattern=args.pattern,
            pytest_args=args.pytest_args,
        )
        retcodes.append(retcode)

    # return the most severe code
    return _most_severe_exit_code(retcodes)


if __name__ == "__main__":
    sys.exit(main())
