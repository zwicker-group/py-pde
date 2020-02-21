'''
Defines a scalar field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import (List, TypeVar, Iterator, Union, Optional,  # @UnusedImport
                    TYPE_CHECKING)
from pathlib import Path

import numpy as np

from .base import DataFieldBase
from ..grids import UnitGrid, CartesianGrid
from ..grids.base import GridBase 
from ..tools.expressions import ScalarExpression


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
    def from_expression(cls, grid: GridBase, expression: str,
                        label: str = None) -> "ScalarField":
        """ create a scalar field on a grid from a given expression
        
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
        expr = ScalarExpression(expression=expression, signature=grid.axes)
        points = {name: grid.cell_coords[..., i]
                  for i, name in enumerate(grid.axes)}
        return cls(grid=grid, data=expr(**points), label=label)
    
    
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

        
    def laplace(self, bc: "BoundariesData",
                out: Optional['ScalarField'] = None,
                label: str = 'laplace') -> 'ScalarField':
        """ apply Laplace operator and return result as a field 
        
        Args:
            bc: Gives the boundary conditions applied to fields that are
                required for calculating the Laplacian.
            out (ScalarField, optional): Optional scalar field to which the 
                result is written.
            label (str, optional): Name of the returned field
            
        Returns:
            ScalarField: the result of applying the operator 
        """
        if out is not None:
            assert isinstance(out, ScalarField)
        laplace = self.grid.get_operator('laplace', bc=bc)
        return self.apply(laplace, out=out, label=label)
    
    
#     def solve_poisson(self, out: Optional['ScalarField']=None,
#                       label: str="solution to Poisson's equation"):
#         r""" solve Poisson's equation with the current field as inhomogeneity.
#         
#         Denoting the current field by :math:`x`, we thus solve for :math:`y`,
#         defined by the equation 
# 
#         .. math::
#             \nabla^2 y(\boldsymbol r) = -x(\boldsymbol r)
#             
#             
#         Args:
#             out (ScalarField, optional): Optional scalar field to which the 
#                 result is written.
#             label (str, optional): Name of the returned field
#             
#         Returns:
#             ScalarField: the result of applying the operator 
#         """
#         solve_poisson = self.grid.get_operator('poisson_solver',
#                                                bc='periodic')
#         data = solve_poisson(self.data)
#         
#         if out is None:
#             return ScalarField(self.grid, data, label=label)
#         else:
#             out.data = data
#             if label:
#                 out.label = label
#             return out
    
        
    def gradient(self, bc: "BoundariesData",
                 out: Optional['VectorField'] = None,
                 label: str = 'gradient') -> 'VectorField':
        """ apply gradient operator and return result as a field 
        
        Args:
            bc: Gives the boundary conditions applied to fields that are
                required for calculating the gradient.
            out (VectorField, optional): Optional vector field to which the 
                result is written.
            label (str, optional): Name of the returned field
            
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

        
    @property
    def integral(self) -> float:
        """ float: integral of the scalar field over space """
        return self.grid.integrate(self.data)

        
    def to_scalar(self, scalar: Union[str, int] = 'abs',
                  label: Optional[str] = None) -> "ScalarField":
        """ return a modified scalar field by applying `method`
        
        Args:
            scalar (str or int): For scalar fields, only `abs` is supported.
            label (str, optional): Name of the returned field
            
        Returns:
            ScalarField: the scalar result
        """
        if scalar == 'abs' or scalar == 'norm':
            data = np.abs(self.data)
        elif scalar == 'squared_sum':
            data = self.data**2
        else:
            raise ValueError(f'Unknown method `{scalar}` for `to_scalar`')
        return ScalarField(grid=self.grid, data=data, label=label)

               
