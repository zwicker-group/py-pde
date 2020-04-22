'''
Defines a vectorial field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import (Callable, Optional, Union, Any, Dict, List, Sequence,
                    TYPE_CHECKING)

import numpy as np
import numba as nb
        

from .base import DataFieldBase
from .scalar import ScalarField
from ..grids.base import GridBase, DimensionError
from ..tools.numba import jit
from ..tools.docstrings import fill_in_docstring

  
if TYPE_CHECKING:
    from ..grids.boundaries.axes import BoundariesData  # @UnusedImport
    from .tensorial import Tensor2Field  # @UnusedImport



class VectorField(DataFieldBase):
    """ Single vector field on a grid
    
    Attributes:
        grid (:class:`~pde.grids.base.GridBase`):
            The underlying grid defining the discretization
        data (:class:`numpy.ndarray`):
            Vector components at the support points of the grid
        label (str):
            Name of the field
    """
    
    rank = 1
    
    
    @classmethod
    def from_scalars(cls, fields: List[ScalarField], label: str = None) \
            -> "VectorField":
        """ create a vector field from a list of ScalarFields
        
        Note that the data of the scalar fields is copied in the process
        
        Args:
            fields (list):
                The list of (compatible) scalar fields
            label (str, optional):
                Name of the returned field
            
        Returns:
            :class:`VectorField`: the resulting vector field
        """
        grid = fields[0].grid
        
        if grid.dim != len(fields):
            raise DimensionError('Grid dimension and number of scalar fields '
                                 f'differ ({grid.dim} != {len(fields)})')
            
        data = []
        for field in fields:
            assert field.grid.compatible_with(grid)
            data.append(field.data)
            
        return cls(grid, data, label)
    

    @classmethod
    @fill_in_docstring
    def from_expression(cls, grid: GridBase, expressions: Sequence[str],
                        label: str = None) -> "VectorField":
        """ create a vector field on a grid from given expressions

        Warning:
            {WARNING_EXEC}
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            expressions (list of str):
                A list of mathematical expression, one for each component of the
                vector field. The expressions determine the values as a function
                of the position on the grid. The expressions may contain
                standard mathematical functions and they may depend on the axes
                labels of the grid.
            label (str, optional):
                Name of the field
        """
        from ..tools.expressions import ScalarExpression
        
        if isinstance(expressions, str) or len(expressions) != grid.dim:
            axes_names = grid.axes + grid.axes_symmetric
            raise ValueError(f'Expected {grid.dim} expressions for the '
                             f'coordinates {axes_names}.') 
        
        # obtain the coordinates of the grid points
        points = {name: grid.cell_coords[..., i]
                  for i, name in enumerate(grid.axes)}
        
        # evaluate all vector components at all points 
        data = []
        for expression in expressions:
            expr = ScalarExpression(expression=expression, signature=grid.axes)
            values = np.broadcast_to(expr(**points), grid.shape) 
            data.append(values)

        # create vector field from the data
        return cls(grid=grid,  # lgtm [py/call-to-non-callable]
                   data=data,
                   label=label)
        

    def __getitem__(self, key: int) -> ScalarField:
        """ extract a component of the VectorField """
        if not isinstance(key, int):
            raise IndexError('Index must be an integer')
        return ScalarField(self.grid, self.data[key]) 
                           
     
    def dot(self, other: Union["VectorField", "Tensor2Field"],
            out: Optional[Union[ScalarField, "VectorField"]] = None,
            label: str = 'dot product') -> Union[ScalarField, "VectorField"]:
        """ calculate the dot product involving a vector field
        
        This supports the dot product between two vectors fields as well as the
        product between a vector and a tensor. The resulting fields will be a
        scalar or vector, respectively.
        
        Args:
            other (VectorField or Tensor2Field):
                the second field
            out (ScalarField or VectorField, optional):
                Optional field to which the  result is written.
            label (str, optional):
                Name of the returned field
            
        Returns:
            ScalarField or VectorField: the result of applying the dot operator             
        """
        from .tensorial import Tensor2Field  # @Reimport
        
        # check input
        self.grid.assert_grid_compatible(other.grid)
        if isinstance(other, VectorField):
            result_type = ScalarField
        elif isinstance(other, Tensor2Field):
            result_type = VectorField  # type: ignore
        else:
            raise TypeError('Second term must be a vector or tensor field')

        if out is None:
            out = result_type(self.grid)
        else:
            assert isinstance(out, result_type)
            self.grid.assert_grid_compatible(out.grid)
            
        # calculate the result
        np.einsum('i...,i...->...', self.data, other.data, out=out.data)
        if label is not None:
            out.label = label
            
        return out
    
    __matmul__ = dot  # support python @-syntax for matrix multiplication
    
    
    def get_dot_operator(self) -> Callable:
        """ return operator calculating the dot product involving vector fields
        
        This supports both products between two vectors as well as products
        between a vector and a tensor.
        
        Warning:
            This function does not check types or dimensions.
        
        Returns:
            function that takes two instance of :class:`numpy.ndarray`, which
            contain the discretized data of the two operands. An optional third
            argument can specify the output array to which the result is
            written. Note that the returned function is jitted with numba for
            speed.
        """
        dim = self.grid.dim
        
        @jit
        def inner(a, b, out):
            """ calculate dot product between fields `a` and `b` """
            out[:] = a[0] * b[0]  # overwrite potential data in out
            for i in range(1, dim):
                out[:] += a[i] * b[i]
            return out
        
        if nb.config.DISABLE_JIT:
            def dot(a: np.ndarray, b: np.ndarray, out: np.ndarray = None) \
                    -> np.ndarray:
                """ wrapper deciding whether the underlying function is called
                with or without `out`. """
                if out is None:
                    out = np.empty(b.shape[1:])
                return inner(a, b, out)
            
        else:
            @nb.generated_jit
            def dot(a: np.ndarray, b: np.ndarray, out: np.ndarray = None) \
                    -> np.ndarray:
                """ wrapper deciding whether the underlying function is called
                with or without `out`. """
                if isinstance(a, nb.types.Number):
                    # simple scalar call -> do not need to allocate anything
                    raise RuntimeError('Dot needs to be called with fields')
                
                elif isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                    # function is called without `out`
                    def f_with_allocated_out(a, b, out):
                        """ helper function allocating output array """
                        return inner(a, b, out=np.empty(b.shape[1:]))
                
                    return f_with_allocated_out
                
                else:
                    # function is called with `out` argument
                    return inner          
        
        return dot
    
    
    def outer_product(self, other: "VectorField", out: "Tensor2Field" = None,
                      label: str = None) -> "Tensor2Field":
        """ calculate the outer product of this vector field with another
        
        Args:
            other (:class:`VectorField`):
                The second vector field
            out (:class:`pde.fields.tensorial.Tensor2Field`, optional):
                Optional tensorial field to which the  result is written.
            label (str, optional):
                Name of the returned field
        
        """
        from .tensorial import Tensor2Field  # @Reimport
        self.assert_field_compatible(other)
        
        if out is None:
            out = Tensor2Field(self.grid)
        else:
            self.grid.assert_grid_compatible(out.grid)
            
        # calculate the result
        np.einsum('i...,j...->ij...', self.data, other.data, out=out.data)
        if label is not None:
            out.label = label
                
        return out
        

    @fill_in_docstring
    def divergence(self, bc: "BoundariesData",
                   out: Optional[ScalarField] = None,
                   label: str = 'divergence') -> ScalarField:
        """ apply divergence operator and return result as a field 
        
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
        divergence = self.grid.get_operator('divergence', bc=bc)
        if out is None:
            out = ScalarField(self.grid, divergence(self.data), label=label)
        else:
            assert isinstance(out, ScalarField)
            self.grid.assert_grid_compatible(out.grid)
            divergence(self.data, out=out.data)
        return out
    
    
    @fill_in_docstring
    def gradient(self, bc: "BoundariesData",
                 out: Optional['Tensor2Field'] = None,
                 label: str = 'gradient') -> 'Tensor2Field':
        """ apply (vecotr) gradient operator and return result as a field 
        
        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (Tensor2Field, optional):
                Optional tensorial field to which the  result is written.
            label (str, optional):
                Name of the returned field
            
        Returns:
            Tensor2Field: the result of applying the operator 
        """
        vector_gradient = self.grid.get_operator('vector_gradient', bc=bc)
        if out is None:
            from .tensorial import Tensor2Field  # @Reimport
            out = Tensor2Field(self.grid, vector_gradient(self.data),
                               label=label)
        else:
            assert isinstance(out, VectorField)
            self.grid.assert_grid_compatible(out.grid)
            vector_gradient(self.data, out=out.data)
        return out

        
    @fill_in_docstring
    def laplace(self, bc: "BoundariesData",
                out: Optional['VectorField'] = None,
                label: str = 'vector laplacian') -> 'VectorField':
        """ apply vector Laplace operator and return result as a field 
        
        Args:
            bc: 
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (VectorField, optional):
                Optional vector field to which the  result is written.
            label (str, optional):
                Name of the returned field
            
        Returns:
            VectorField: the result of applying the operator 
        """
        if out is not None:
            assert isinstance(out, VectorField)
        laplace = self.grid.get_operator('vector_laplace', bc=bc)
        return self.apply(laplace, out=out, label=label)
    
        
    @property
    def integral(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: integral of each component over space """
        return self.grid.integrate(self.data)
                
    
    def to_scalar(self, scalar: Union[str, int] = 'norm',
                  label: Optional[str] = 'scalar `{scalar}`') -> ScalarField:
        """ return a scalar field by applying `method`
        
        The two tensor invariants are given by
        
        Args:
            scalar (str or int):
                Choose the method to use. Possible  choices are `norm`, `max`,
                `min`, `squared_sum`, and an integer which specifies a
                particular component.
            label (str, optional):
                Name of the returned field
            
        Returns:
            ScalarField: the scalar result
        """
        if scalar == 'norm':
            data = np.linalg.norm(self.data, axis=0)
        elif scalar == 'max':
            data = np.max(self.data, axis=0)
        elif scalar == 'min':
            data = np.min(self.data, axis=0)
        elif scalar == 'squared_sum' or scalar == 'norm_squared':
            data = np.sum(self.data**2, axis=0)
        elif isinstance(scalar, str):
            raise ValueError(f'Unknown method `{scalar}` for `to_scalar`')
        else:
            # assume that `scalar` is an index to extract a component
            data = self.data[scalar]
            
        if label is not None:
            label = label.format(scalar=scalar)
            
        return ScalarField(self.grid, data, label=label)

    
    def get_line_data(self, scalar: Union[str, int] = 'norm',  # type: ignore
                      extract: str = 'auto') -> Dict[str, Any]:
        """ return data for a line plot of the field
        
        Args:
            method (str or int):
                The method for extracting scalars as described in
                `self.to_scalar`.
                
        Returns:
            dict: Information useful for performing a line plot of the field
        """
        return self.to_scalar(scalar=scalar).get_line_data(extract=extract)
                    
    
    def get_image_data(self, scalar: str = 'norm', **kwargs) -> Dict[str, Any]:
        r""" return data for plotting an image of the field

        Args:
            scalar (str or int):
                The method for extracting scalars as described in
                `self.to_scalar`.
            \**kwargs: Additional parameters are forwarded to `get_image_data`
                
        Returns:
            dict: Information useful for plotting an image of the field
        """
        return self.to_scalar(scalar=scalar).get_image_data(**kwargs)
    
    
    def get_vector_data(self, **kwargs) -> Dict[str, Any]:
        r""" return data for a vector plot of the field
        
        Args:
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`
        
        Returns:
            dict: Information useful for plotting an vector field        
        """
        # TODO: Handle Spherical and Cartesian grids, too. This could be
        # implemented by adding a get_vector_data method to the grids
        if self.grid.dim == 2:
            vx = self[0].get_image_data(**kwargs)
            vy = self[1].get_image_data(**kwargs)
            vx['data_x'] = vx.pop('data')
            vx['data_y'] = vy['data']
            vx['title'] = self.label
            return vx
            
        else:
            raise NotImplementedError()
    
