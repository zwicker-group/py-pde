# AGENTS.md

## Project Structure

This repository contains `py-pde`, a Python package for solving partial differential equations (PDEs) on grids using finite differences and optional acceleration backends such as `numba`, `jax`, and `torch`.

Key directories:

- `pde/`: main package source code.
  - `pde/backends/`: backend implementations and operator registration.
  - `pde/fields/`: scalar, vector, and tensor field classes.
  - `pde/grids/`: grid definitions and discretization logic.
  - `pde/pdes/`: PDE model definitions and equation building blocks.
  - `pde/solvers/`: time integration and solver implementations.
  - `pde/storage/`: storage classes and serialization helpers.
  - `pde/tools/`: general utilities.
  - `pde/trackers/`: runtime monitoring and tracking features.
  - `pde/visualization/`: plotting and visualization support.
- `tests/`: automated tests organized by feature area.
- `examples/`: user-facing example scripts and notebooks.
- `docs/`: Sphinx documentation, source pages, and built docs artifacts.
- `scripts/`: project automation, test runners, and maintenance scripts.
- `README.md`: high-level project overview and usage examples.
- `pyproject.toml`: packaging metadata, dependencies, formatting, linting, and test configuration.

When making changes, favor the existing module layout and keep new functionality aligned with the relevant subpackage rather than creating ad hoc top-level modules.

## Coding Conventions

- Write Python code compatible with the supported versions declared in `pyproject.toml` (currently Python >=3.10 and <3.15).
- Prefer idiomatic, readable Python over clever one-liners.
- Public methods and functions should have docstrings documenting arguments and behavior; the first line should summarize the purpose.
- Maintain the project’s existing naming conventions and type usage patterns.
- Keep backend-specific code inside the relevant `pde/backends/*` modules and avoid duplicating logic across backends unless necessary.
- Follow the repository’s linting and formatting setup:
  - `ruff` for linting and import sorting.
  - `black` formatting conventions.
  - `isort` ordering consistent with the project config.
- Keep imports organized by standard library, third-party, then first-party modules.
- Use explicit type annotations where the project already expects them, especially in public APIs and backend interfaces.
- Avoid introducing unnecessary dependencies or optional packages without a clear project need.
- If a change affects numerical behavior, include or update focused tests rather than relying only on manual inspection.

## Quality Gates

Before considering a change complete, verify the relevant behavior with the smallest focused checks available.

Common validation commands:

- Run the relevant unit tests for the modified area, for example:
  - `pytest tests/...`
  - or the project helper scripts under `scripts/` when they target a broader validation path.
- Run linting and formatting checks when changing Python files, especially if the change touches imports, syntax, or conventions:
  - `ruff check .`
  - `ruff format --check .`
- Type checking is part of the project’s quality expectations where relevant:
  - `mypy pde`
- If a bug fix or numerical change is involved, prefer a failing test reproducing the issue before the fix, then verify the same test passes after the fix.

The project is test-driven and numerical correctness matters; do not claim a fix is complete without fresh verification evidence.

## Project Constraints

- The library is scientific/numerical software, so correctness and stability matter more than short-term convenience.
- Backends may be compiled or JIT-optimized (`numba`, `jax`, `torch`), so code must remain compatible with backend restrictions and traceability constraints.
- Some repository patterns explicitly discourage unsafe JIT behavior, such as boolean indexing or traced control flow in JAX/Numba paths; keep numerical kernels backend-safe.
- Optional dependencies are intentionally split by feature area (`io`, `interactive`, `mpi`), and code should avoid hard requirements on optional packages unless the feature truly depends on them.
- The project targets research and educational workflows, so APIs should remain clear and easy to use while still supporting performance-sensitive execution.
- Changes should be compatible with the supported Python version range and avoid features that require newer language versions than the project declares.
- Keep documentation and examples aligned with behavior changes when APIs or numerical semantics are affected.

## Working Rules for Agents

- Stay within the existing architecture and submodule boundaries.
- Prefer minimal, targeted edits over broad refactors.
- Keep patch scope consistent with the issue or task being solved.
- If there is uncertainty about a backend-specific behavior or a project pattern, inspect the relevant module and neighboring implementations before changing code.
- Document important assumptions or constraints in code comments only when they materially improve maintainability.
