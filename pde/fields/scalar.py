'''
Defines a scalar field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Union, Sequence, Dict, Optional, TYPE_CHECKING
from pathlib import Path

import numpy as np

from .base import DataFieldBase
from ..grids import UnitGrid, CartesianGrid
from ..grids.base import GridBase, DomainError
from ..tools.docstrings import fill_in_docstring


if TYPE_CHECKING:
    from ..grids.boundaries.axes import BoundariesData  # @UnusedImport
    from .vectorial import VectorField  # @UnusedImport



class ScalarField(DataFieldBase):
    """ Single scalar field on a grid

    Attributes:
        grid (:class:`~pde.grids.GridBase`):
            The underlying grid defining the discretization
        data (:class:`np.ndarray`):
            Scalar values at the support points of the grid
        label (str):
            Name of the field
    """

    rank = 0
        
        
    @classmethod
    @fill_in_docstring
    def from_expression(cls, grid: GridBase, expression: str,
                        label: str = None) -> "ScalarField":
        """ create a scalar field on a grid from a given expression
        
        Warning:
            {WARNING_EXEC}
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            expression (str):
                Mathematical expression for the scalar value as a function of
                the position on the grid. The expression may contain standard
                mathematical functions and it may depend on the axes labels of
                the grid.
            label (str, optional):
                Name of the field
        """
        from ..tools.expressions import ScalarExpression
        expr = ScalarExpression(expression=expression, signature=grid.axes)
        points = {name: grid.cell_coords[..., i]
                  for i, name in enumerate(grid.axes)}
        return cls(grid=grid,  # lgtm [py/call-to-non-callable]
                   data=expr(**points),
                   label=label)
    
    
    @classmethod
    def from_image(cls, path: Union[Path, str], bounds=None, periodic=False,
                   label: str = None) -> "ScalarField":
        """ create a scalar field from an image
    
        Args:
            path (:class:`Path` or str): The path to the image
            bounds (tuple, optional): Gives the coordinate range for each axis.
                This should be two tuples of two numbers each, which mark the
                lower and upper bound for each axis. 
            periodic (bool or list): Specifies which axes possess periodic
                boundary conditions. This is either a list of booleans defining
                periodicity for each individual axis or a single boolean value
                specifying the same periodicity for all axes.
            label (str, optional):
                Name of the field
        """
        from matplotlib.pyplot import imread
        # read image and convert to grayscale
        data = imread(path)
        if data.ndim == 2:
            pass  # is already gray scale
        elif data.ndim == 3:
            # convert to gray scale using ITU-R 601-2 luma transform:
            weights = np.array([0.299, 0.587, 0.114])
            data = data[..., :3] @ weights
        else:
            raise RuntimeError(f'Image data has wrong shape: {data.shape}')
        
        # transpose data to use mathematical conventions for axes
        data = data.T[:, ::-1]
        
        # determine the associated grid
        if bounds is None:
            grid: GridBase = UnitGrid(data.shape, periodic=periodic)
        else:
            grid = CartesianGrid(bounds, data.shape, periodic=periodic)
        
        return cls(grid, data, label=label)
        
     
    @DataFieldBase._data_flat.setter  # type: ignore
    def _data_flat(self, value):
        """ set the data from a value from a collection """
        self._data = value[0]

        
    @fill_in_docstring
    def laplace(self, bc: "BoundariesData",
                out: Optional['ScalarField'] = None,
                label: str = 'laplace') -> 'ScalarField':
        """ apply Laplace operator and return result as a field 
        
        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (ScalarField, optional):
                Optional scalar field to which the  result is written.
            label (str, optional):
                Name of the returned field
            
        Returns:
            ScalarField: the result of applying the operator 
        """
        if out is not None:
            assert isinstance(out, ScalarField)
        laplace = self.grid.get_operator('laplace', bc=bc)
        return self.apply(laplace, out=out, label=label)

        
    @fill_in_docstring
    def gradient(self, bc: "BoundariesData",
                 out: Optional['VectorField'] = None,
                 label: str = 'gradient') -> 'VectorField':
        """ apply gradient operator and return result as a field 
        
        Args:
            bc: 
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (VectorField, optional):
                Optional vector field to which the result is written.
            label (str, optional):
                Name of the returned field
            
        Returns:
            VectorField: the result of applying the operator 
        """
        from .vectorial import VectorField  # @Reimport
        
        gradient = self.grid.get_operator('gradient', bc=bc)
        if out is None:
            out = VectorField(self.grid, gradient(self.data), label=label)
        else:
            assert isinstance(out, VectorField)
            gradient(self.data, out=out.data)
        return out
    
    
    @fill_in_docstring
    def solve_poisson(self, bc: "BoundariesData",
                      out: Optional['ScalarField'] = None,
                      label: str = "Solution to Poisson's equation"):
        r""" solve Poisson's equation with the current field as inhomogeneity.
         
        Denoting the current field by :math:`x`, we thus solve for :math:`y`,
        defined by the equation 
 
        .. math::
            \nabla^2 y(\boldsymbol r) = -x(\boldsymbol r)
            
        with boundary conditions specified by `bc`.
            
        Note:
            In case of periodic or Neumann boundary conditions, the right hand
            side :math:`x(\boldsymbol r)` needs to satisfy the following
            condition for consistency:
            
            .. math::
                \int x \, \mathrm{d}V = \oint g \, \mathrm{d}S
                
            where :math:`g` denotes the function specifying the outwards
            derivative for Neumann conditions. In particular, the integral over
            :math:`x` must vanish for neutral Neumann or periodic conditions.
             
        Args:
            bc: 
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (ScalarField, optional):
                Optional scalar field to which the  result is written.
            label (str, optional):
                Name of the returned field
             
        Returns:
            ScalarField: the result of applying the operator 
        """
        # Deprecated this method on 2020-04-15
        import warnings
        warnings.warn("solve_poisson() method is deprecated. Use the function "
                      "pde.pdes.solve_poisson_equation or pde.pdes.solve_"
                      "laplace_equation instead.",
                      DeprecationWarning)
        
        # solve the poisson problem
        solve_poisson = self.grid.get_operator('poisson_solver', bc=bc)
        try:
            result = solve_poisson(self.data)
        except RuntimeError:
            average = self.average
            if abs(average) > 1e-10:
                raise RuntimeError('Could not solve the Poisson problem. One '
                                   'possible reason for this is that only '
                                   'periodic or Neumann conditions are '
                                   'applied although the average of the field '
                                   f'is {average} and thus non-zero.')
            else:
                raise  # another error occured
         
        if out is None:
            return ScalarField(self.grid, result, label=label)
        else:
            out.data = result
            if label:
                out.label = label
            return out
    
        
    @property
    def integral(self) -> float:
        """ float: integral of the scalar field over space """
        return float(self.grid.integrate(self.data))

               
    def project(self, axes: Union[str, Sequence[str]],
                method: str = 'integral',
                label: str = None) -> "ScalarField":
        """ project scalar field along given axes
        
        Args:
            axes (list of str):
                The names of the axes that are removed by the projection
                operation. The valid names for a given grid are the ones in
                the :attr:`GridBase.axes` attribute.
            method (str):
                The projection method. This can be either 'integral' to
                integrate over the removed axes or 'average' to perform an
                average instead.
            label (str, optional):
                The label of the returned field
                
        Returns:
            ScalarField: The projected data in a scalar field with a subgrid of
            the original grid.
        """
        if any(ax not in self.grid.axes for ax in axes):
            raise ValueError(f'The axes {axes} are not all contained in '
                             f'{self.grid} with axes {self.grid.axes}')
            
        # determine the axes after projection
        ax_all = range(self.grid.num_axes)
        ax_remove = tuple(self.grid.axes.index(ax) for ax in axes)
        ax_retain = tuple(sorted(set(ax_all) - set(ax_remove)))
        
        # determine the new grid
        subgrid = self.grid.get_subgrid(ax_retain)
        
        # calculate the new data
        if method == 'integral':
            subdata = self.grid.integrate(self.data, axes=ax_remove)
        elif method == 'average' or method == 'mean':
            subdata = (self.grid.integrate(self.data, axes=ax_remove) /
                       self.grid.integrate(1, axes=ax_remove))
        else:
            raise ValueError(f'Unknown projection method `{method}`')
        
        # create the new field instance
        return self.__class__(grid=subgrid, data=subdata, label=label)
    
    
    def slice(self, position: Dict[str, float],
              method: str = 'nearest',
              label: str = None) -> "ScalarField":
        """ slice data at a given position
        
        Args:
            position (dict):
                Determines the location of the slice using a dictionary
                supplying coordinate values for a subset of axes. Axes not
                mentioned in the dictionary are retained and form the slice.
                For instance, in a 2d Cartesian grid, `position = {'x': 1}`
                slices along the y-direction at x=1. Additionally, the special
                positions 'low', 'mid', and 'high' are supported to reference
                relative positions along the axis.
            method (str):
                The method used for slicing. `nearest` takes data from cells
                defined on the grid.
            label (str, optional):
                The label of the returned field
                
        Returns:
            ScalarField: The sliced data in a scalar field with a subgrid of
            the original grid.
        """
        grid = self.grid
        
        # parse the positions and determine the axes to remove
        ax_remove, pos_values = [], np.zeros(grid.num_axes)
        for ax, pos in position.items():
            # check the axis
            try:
                i = grid.axes.index(ax)
            except ValueError:
                raise ValueError(f'The axes {ax} is not contained in '
                                 f'{self.grid} with axes {self.grid.axes}')
            ax_remove.append(i)
            
            # check the position
            if isinstance(pos, str):
                if pos in {'min', 'low', 'lower'}:
                    pos_values[i] = grid.axes_coords[i][0]
                elif pos in {'max', 'high', 'upper'}:
                    pos_values[i] = grid.axes_coords[i][-1]
                elif pos in {'mid', 'middle', 'center'}:
                    pos_values[i] = np.mean(grid.axes_bounds[i])
                else:
                    raise ValueError(f'Unknown position `{pos}`')
            else:
                pos_values[i] = float(pos)
            
        # determine the axes left after slicing and the new grid
        ax_all = range(grid.num_axes)
        ax_retain = tuple(sorted(set(ax_all) - set(ax_remove)))
        subgrid = grid.get_subgrid(ax_retain)
        
        # obtain the sliced data
        if method == 'nearest':
            idx = []
            for i in range(grid.num_axes):
                if i in ax_remove:
                    pos = pos_values[i]
                    axis_bounds = grid.axes_bounds[i]
                    if pos < axis_bounds[0] or pos > axis_bounds[1]:
                        raise DomainError(f'Position {grid.axes[i]} = {pos} is '
                                          'outside the domain')
                    # add slice that is closest to pos 
                    idx.append(np.argmin((grid.axes_coords[i] - pos)**2))
                else:
                    idx.append(slice(None))
            subdata = self.data[tuple(idx)]
            
        else:
            raise ValueError(f'Unknown slicing method `{method}`')
    
        # create the new field instance
        return self.__class__(grid=subgrid, data=subdata, label=label)
    
        
    def to_scalar(self, scalar: str = 'auto',
                  label: Optional[str] = None) -> "ScalarField":
        """ return a modified scalar field by applying `method`
        
        Args:
            scalar (str or int):
                How to obtain the scalar. For ScalarField, the default `0`
                simply returns the actual field. Setting this to 'norm' returns
                the absolute value at each point.
            label (str, optional): Name of the returned field
            
        Returns:
            :class:`pde.fields.scalar.ScalarField`: the scalar field after
            applying the operation
        """
        if scalar == 'auto':
            data = self.data
            
        elif scalar == 'abs' or scalar == 'norm':
            data = np.abs(self.data)
            
        elif scalar == 'squared_sum':
            data = self.data**2
            
        else:
            raise ValueError(f'Unknown method `{scalar}` for `to_scalar`')
        
        return ScalarField(grid=self.grid, data=data, label=label)

