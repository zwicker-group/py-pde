"""Methods for automatic transformation of docstrings.

.. autosummary::
   :nosignatures:

   get_text_block
   replace_in_docstring
   fill_in_docstring

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import re
import textwrap
from functools import partial
from typing import Callable, TypeVar

DOCSTRING_REPLACEMENTS = {
    # description of function arguments
    "ARG_BOUNDARIES_INSTANCE": """
        Specifies the boundary conditions applied to the field. This must be an
        instance of :class:`~pde.grids.boundaries.axes.BoundariesList`, which can be
        created from various data formats using the class method
       :meth:`~pde.grids.boundaries.axes.BoundariesBase.from_data`.
       """,
    "ARG_BOUNDARIES": """
        Boundary conditions are generally given as a dictionary with one condition for
        each axis side. For periodic axes, only periodic boundary conditions are allowed
        (indicated by 'periodic' and 'anti-periodic'). For non-periodic axes, different
        boundary conditions can be specified for the lower and upper end (using specific
        identifiers, like `x-` and `y+`). For instance, Dirichlet conditions enforcing a
        value NUM (specified by `{'value': NUM}`) and Neumann conditions enforcing the
        value DERIV for the derivative in the normal direction (specified by
        `{'derivative': DERIV}`) are supported. Note that the special value
        'auto_periodic_neumann' imposes periodic boundary conditions for periodic axis
        and a vanishing derivative otherwise. More information can be found in the
        :ref:`boundaries documentation <documentation-boundaries>`.
        """,
    "ARG_TRACKER_INTERRUPT": """
        Determines when the tracker interrupts the simulation. A single numbers
        determines an interval (measured in the simulation time unit) of regular
        interruption. A string is interpreted as a duration in real time assuming the
        format 'hh:mm:ss'. A list of numbers is taken as explicit simulation time points.
        More fine-grained control is possible by passing an instance of classes defined
        in :mod:`~pde.trackers.interrupts`.
        """,
    "ARG_PLOT_QUANTITIES": """
        A 2d list of quantities that are shown in a rectangular arrangement.
        If `quantities` is a simple list, the panels will be rendered as a
        single row.
        Each panel is defined by a dictionary, where the mandatory item 'source'
        defines what is being shown. Here, an integer specifies the component
        that is extracted from the field while a function is evaluate with the
        full state as an input and the result is shown.
        Additional items in the dictionary can be 'title' (setting the title of
        the panel), 'scale' (defining the color range shown; these are typically
        two numbers defining the lower and upper bound, but if only one is given
        the range [0, scale] is assumed), and 'cmap' (defining the colormap
        being used).
        """,
    "ARG_PLOT_SCALE": """
        Flag determining how the range of the color scale is determined. In the
        simplest case a tuple of numbers marks the lower and upper end of the 
        scalar values that will be shown. If only a single number is supplied,
        the range starts at zero and ends at the given number. Additionally, the
        special value 'automatic' determines the range from the range of scalar
        values.
        """,
    # descriptions of the discretization and the symmetries
    "DESCR_CYLINDRICAL_GRID": r"""
        The cylindrical grid assumes polar symmetry, so that fields only depend
        on the radial coordinate `r` and the axial coordinate `z`. Here, the
        first axis is along the radius, while the second axis is along the axis
        of the cylinder. The radial discretization is defined as
        :math:`r_i = (i + \frac12) \Delta r` for :math:`i=0, \ldots, N_r-1`.
        """,
    "DESCR_POLAR_GRID": r"""
        The polar grid assumes polar symmetry, so that fields only depend on the
        radial coordinate `r`. The radial discretization is defined as
        :math:`r_i = r_\mathrm{min} + (i + \frac12) \Delta r` for
        :math:`i=0, \ldots, N_r-1`,  where :math:`r_\mathrm{min}` is the radius
        of the inner boundary, which is zero by default. Note that the radius of
        the outer boundary is given by
        :math:`r_\mathrm{max} = r_\mathrm{min} + N_r \Delta r`.
        """,
    "DESCR_SPHERICAL_GRID": r"""
        The spherical grid assumes spherical symmetry, so that fields only
        depend on the radial coordinate `r`. The radial discretization is
        defined as :math:`r_i = r_\mathrm{min} + (i + \frac12) \Delta r` for
        :math:`i=0, \ldots, N_r-1`,  where :math:`r_\mathrm{min}` is the radius
        of the inner boundary, which is zero by default. Note that the radius of
        the outer boundary is given by
        :math:`r_\mathrm{max} = r_\mathrm{min} + N_r \Delta r`.
        """,
    # notes in the docstring
    "WARNING_EXEC": r"""
        This implementation uses :func:`exec` and should therefore not be used 
        in a context where malicious input could occur.
        """,
}
DOCSTRING_REPLACEMENTS["ARG_BOUNDARIES_OPTIONAL"] = (
    DOCSTRING_REPLACEMENTS["ARG_BOUNDARIES"]
    + "If the special value `None` is given, no boundary conditions are enforced. The "
    "user then needs to ensure that the ghost cells are set accordingly. "
)
DOCSTRING_REPLACEMENTS = {k: v[1:-1] for k, v in DOCSTRING_REPLACEMENTS.items()}


def get_text_block(identifier: str) -> str:
    """Return a single text block.

    Args:
        identifier (str): The name of the text block

    Returns:
        str: the text block as one long line.
    """
    raw_text = DOCSTRING_REPLACEMENTS[identifier]
    return "".join(textwrap.dedent(raw_text))


TFunc = TypeVar("TFunc", bound=Callable)


def replace_in_docstring(
    f: TFunc, token: str, value: str, docstring: str | None = None
) -> TFunc:
    """Replace a text in a docstring using the correct indentation.

    Args:
        f (callable): The function with the docstring to handle
        token (str): The token to search for
        value (str): The replacement string
        docstring (str): A docstring that should be used instead of f.__doc__

    Returns:
        callable: The function with the modified docstring
    """
    # initialize textwrapper for formatting docstring

    def repl(matchobj) -> str:
        """Helper function replacing token in docstring."""
        bare_text = textwrap.dedent(value).strip()
        return textwrap.indent(bare_text, matchobj.group(1))

    if docstring is None:
        docstring = f.__doc__

    # replace the token with the correct indentation
    f.__doc__ = re.sub(
        f"^([ \t]*){token}",
        repl,
        docstring,  # type: ignore
        flags=re.MULTILINE,
    )

    return f


def fill_in_docstring(f: TFunc) -> TFunc:
    """Decorator that replaces text in the docstring of a function."""
    tw = textwrap.TextWrapper(
        width=80, expand_tabs=True, replace_whitespace=True, drop_whitespace=True
    )

    def repl(matchobj, value: str) -> str:
        """Helper function replacing token in docstring."""
        tw.initial_indent = tw.subsequent_indent = matchobj.group(1)
        return tw.fill(textwrap.dedent(value))

    for name, value in DOCSTRING_REPLACEMENTS.items():
        token = "{" + name + "}"
        f.__doc__ = re.sub(
            f"^([ \t]*){token}",
            partial(repl, value=value),
            f.__doc__,  # type: ignore
            flags=re.MULTILINE,
        )
    return f
