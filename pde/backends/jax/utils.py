"""Defines utilities for the jax backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections.abc import Callable

import jax.numpy as jnp

SPECIAL_FUNCTIONS_JAX: dict[str, Callable] = {
    "Heaviside": jnp.heaviside,
    "hypot": jnp.hypot,
}
