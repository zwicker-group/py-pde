'''
Defines a collection of fields to represent multiple fields defined on a common
grid.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import (Sequence, Optional, Union, Any, Dict,
                    List, TypeVar, Iterator)  # @UnusedImport

import numpy as np

from .base import FieldBase, DataFieldBase
from .scalar import ScalarField
from ..grids.base import GridBase



class FieldCollection(FieldBase):
    """ Collection of fields defined on the same grid """


    def __init__(self, fields: Sequence[DataFieldBase],
                 data=None,
                 copy_fields: bool = False,
                 label: Optional[str] = None):
        """ 
        Args:
            fields:
                Sequence of the individual fields
            data (:class:`numpy.ndarray`):
                Data of the fields. If `None`, the data is instead taken from
                the individual fields given by `fields`.
            copy_fields (bool):
                Flag determining whether the individual fields given in `fields`
                are copied. 
            label (str): Label of the field collection
            
        Warning:
            If `data` is given and `copy_fields == False`, the data in the
            individual fields is overwritten by the associated `data`.
        """
        if isinstance(fields, FieldCollection):
            # support assigning a field collection for convenience
            fields = fields.fields
            
        if len(fields) == 0:
            raise ValueError('At least one field must be defined')
        
        # check if grids are compatible
        grid = fields[0].grid
        if any(grid != f.grid for f in fields[1:]):
            grids = [f.grid for f in fields]
            raise RuntimeError(f'Grids are incompatible: {grids}')

        # create the list of underlying fields        
        if copy_fields:
            self.fields = [field.copy() for field in fields]
        else:
            self.fields = fields  # type: ignore
        
        # extract data from individual fields
        new_data: List[np.ndarray] = []
        self.slices: List[slice] = []
        dof = 0  # count local degrees of freedom
        for field in self.fields:
            if not isinstance(field, DataFieldBase):
                raise RuntimeError('Individual fields must be of type '
                                   'DataFieldBase. FieldCollections cannot be '
                                   'nested')
            start = len(new_data)
            this_data = field._data_flat
            new_data.extend(this_data)
            self.slices.append(slice(start, len(new_data)))
            dof += len(this_data)
                
        # combine into one data field
        data_shape = (dof,) + grid.shape
        if data is None:
            data = np.array(new_data, dtype=np.double)
        else:
            data = np.asarray(data, dtype=np.double)
            if data.shape != data_shape:
                data = np.array(np.broadcast_to(data, data_shape))
        assert data.shape == data_shape
        
        # initialize the class
        super().__init__(grid, data, label=label)        
            
        # link the data of the original fields back to self._data if they were
        # not copied
        if not copy_fields:
            for i, field in enumerate(self.fields):
                field_shape = field.data.shape
                field._data_flat = self.data[self.slices[i]]
                
                # check whether the field data is based on our data field
                assert field.data.base is self.data
                assert field.data.shape == field_shape
         
         
    def __repr__(self):
        """ return instance as string """
        fields = []
        for f in self.fields:
            name = f.__class__.__name__
            if f.label:
                fields.append(f'{name}(..., label="{f.label}")')
            else:
                fields.append(f'{name}(...)')
        return f"{self.__class__.__name__}({', '.join(fields)})"

        
    def __len__(self):
        """ return the number of stored fields """
        return len(self.fields)
    
    
    def __iter__(self) -> Iterator[DataFieldBase]:
        """ return iterator over the actual fields """
        return iter(self.fields)
    
    
    def __getitem__(self, index: Union[int, str]) -> DataFieldBase:
        """ return a specific field """
        if isinstance(index, int):
            # simple numerical index
            return self.fields[index]
        
        elif isinstance(index, str):
            # index specifying the label of the field
            for field in self.fields:
                if field.label == index:
                    return field
            raise KeyError(f'No field with name {index}')
        
        else:
            raise TypeError(f'Unsupported index {index}')

        
    def __setitem__(self, index: int, value):
        """ set the value of a specific field """
        # We need to load the field and set data explicitly
        # WARNING: Do not use `self.fields[index] = value`, since this would
        # break the connection between the data fields 
        if isinstance(index, int):
            # simple numerical index
            self.fields[index].data = value
            
        elif isinstance(index, str):
            # index specifying the label of the field
            for field in self.fields:
                if field.label == index:
                    field.data = value
                    break
            else:
                raise KeyError(f'No field with name {index}')

        else:
            raise TypeError(f'Unsupported index {index}')

        
    @classmethod
    def from_state(cls, state: Dict[str, Any], grid: GridBase,  # type: ignore
                   data=None) -> "FieldCollection":
        """ create a field collection from given state.
        
        Args:
            state (str or dict): State from which the instance is created. If 
                `state` is a string, it is decoded as JSON.
            grid (:class:`~pde.grids.GridBase`):
                The grid that is used to describe the field
            data (:class:`numpy.ndarray`, optional): Data values at the support
                points of the grid that define the field.
        """
        fields = [FieldBase.from_state(field_state, grid=grid)
                  for field_state in state.pop('fields')]
        
        return cls(fields, data=data, **state)  # type: ignore


    @classmethod
    def _from_dataset(cls, dataset) -> "FieldCollection":
        """ construct the class by reading data from an hdf5 dataset """
        count = dataset.attrs['count']  # number of fields
        label = dataset.attrs['label'] if 'label' in dataset.attrs else None
        
        # reconstruct all fields
        fields = [FieldBase._from_dataset(dataset[f"field_{i}"])
                  for i in range(count)]
             
        return cls(fields, label=label)  # type: ignore


    def _write_hdf_dataset(self, fp):
        """ write data to a given hdf5 file pointer `fp` """
        # write details of the collection
        fp.attrs['class'] = self.__class__.__name__
        fp.attrs['count'] = len(self)
        if self.label:      
            fp.attrs['label'] = str(self.label)
                  
        # write individual fields
        for i, field in enumerate(self.fields):
            field._write_hdf_dataset(fp, f"field_{i}")


    def assert_field_compatible(self, other: FieldBase,
                                accept_scalar: bool = False):
        """ checks whether `other` is compatible with the current field
        
        Args:
            other (FieldBase): Other field this is compared to
            accept_scalar (bool, optional): Determines whether it is acceptable
                that `other` is an instance of
                :class:`~pde.fields.ScalarField`.
        """
        super().assert_field_compatible(other, accept_scalar=accept_scalar)

        # check whether all sub fields are compatible
        if isinstance(other, FieldCollection):
            for f1, f2 in zip(self, other):
                f1.assert_field_compatible(f2, accept_scalar=accept_scalar)
                

    @classmethod
    def scalar_random_uniform(cls, num_fields: int, grid: GridBase,
                              vmin: float = 0, vmax: float = 1,
                              label: Optional[str] = None):
        """ create scalar fields with random values between `vmin` and `vmax`
        
        Args:
            num_fields (int): The number of fields to create
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which the fields are defined
            vmin (float): Smallest random value
            vmax (float): Largest random value
            label (str, optional): Name of the field collection
        """
        return cls([ScalarField.random_uniform(grid, vmin, vmax)
                    for _ in range(num_fields)], label=label)
    
    
    @property
    def state(self) -> dict:
        """ dict: current state of this instance """
        return {'label': self.label,
                'fields': [f.state for f in self.fields]}


    @property
    def state_serialized(self) -> str:
        """ str: a json serialized version of the field """
        import json
        
        fields = []
        for field in self.fields:
            state = field.state
            state['class'] = field.__class__.__name__
            fields.append(state)

        return json.dumps({'class': self.__class__.__name__,
                           'label': self.label,
                           'fields': fields})
    
    
    def copy(self, data=None, label: str = None) -> 'FieldCollection':
        """ return a copy of the data, but not of the grid
        
        Args:
            data (:class:`numpy.ndarray`, optional): Data values at the support
                points of the grid that define the field. Note that the data is
                not copied but used directly.
            label (str, optional): Name of the copied field
        """
        if label is None:
            label = self.label
        fields = [f.copy() for f in self.fields]
        # if data is None, the data of the individual fields is copied in their
        # copy() method above. The underlying data is therefore independent from
        # the current field 
        return self.__class__(fields, data=data, label=label)


    def interpolate_to_grid(self, grid: GridBase,
                            normalized: bool = False,
                            method: str = 'linear',
                            label: Optional[str] = None) -> 'FieldCollection':
        """ interpolate the data of this field collection to another grid.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            normalized (bool): Specifies whether the interpolation uses the true
                grid positions, potentially filling in undefined locations with
                zeros. Alternatively, if `normalized = True`, the data is
                stretched so both fields cover the same area.
            method (str): Specifies interpolation method, e.g. 'linear' or
                'nearest'
            label (str, optional): Name of the returned field collection
            
        Returns:
            Field collection with the same fields as the current one
        """        
        if label is None:
            label = self.label
        fields = [f.interpolate_to_grid(grid, normalized=normalized,
                                        method=method)
                  for f in self.fields]
        return self.__class__(fields, label=label)        
        

    def smooth(self, sigma: Optional[float] = 1,
               out: Optional['FieldCollection'] = None,
               label: str = None) -> 'FieldCollection':
        """ applies Gaussian smoothing with the given standard deviation

        This function respects periodic boundary conditions of the underlying
        grid, using reflection when no periodicity is specified.
        
        sigma (float, optional):
            Gives the standard deviation of the smoothing in real length units
            (default: 1)
        out (FieldCollection, optional):
            Optional field into which the smoothed data is stored
        label (str, optional):
            Name of the returned field

        Returns:
            Field collection with smoothed data, stored at `out` if given.             
        """
        # allocate memory for storing output
        if out is None:
            out = self.copy(label=label)
        else:
            self.assert_field_compatible(out)
            if label:
                out.label = label
         
        # apply Gaussian smoothing for each axis
        for f_in, f_out in zip(self, out):        
            f_in.smooth(sigma=sigma, out=f_out)
             
        return out   
         
        
    def get_line_data(self, index: int = 0,  # type: ignore
                      **kwargs) -> Dict[str, Any]:
        r""" return data for a line plot of the field
        
        Args:
            index (int): Index of the field whose data is returned
            \**kwargs: Arguments forwarded to the `get_line_data` method
                
        Returns:
            dict: Information useful for performing a line plot of the field
        """
        return self[index].get_line_data(**kwargs)
    
    
    def get_image_data(self, index: int = 0, **kwargs) -> Dict[str, Any]:
        r""" return data for plotting an image of the field

        Args:
            index (int): Index of the field whose data is returned
            \**kwargs: Arguments forwarded to the `get_image_data` method
                
        Returns:
            dict: Information useful for plotting an image of the field
        """
        return self[index].get_image_data(**kwargs)
    
    
    def plot_collection(self,
                        title: Optional[str] = None,
                        show: bool = False,
                        **kwargs):
        r""" visualize all fields by plotting them next to each other
        
        Args:
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Additional keyword arguments are passed to the method
                `plot_line` of the individual fields.
        """
        import matplotlib.pyplot as plt
        _, axes = plt.subplots(1, len(self), figsize=(4 * len(self), 3))
        for field, ax in zip(self.fields, axes):
            field.plot(ax=ax, **kwargs)
        if title is None:
            title = self.label
        plt.suptitle(title)
        if show:
            plt.show()
    
    
    def plot_line(self,
                  title: Optional[str] = None,
                  show: bool = False,
                  **kwargs):
        r""" visualize all fields using a 1d cuts
        
        Args:
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Additional keyword arguments are passed to the method
                `plot_line` of the individual fields.
        """
        self.plot_collection(title=title, show=show, kind='line', **kwargs)
        
    
    def plot_image(self, quantities=None,  # type: ignore
                   title: Optional[str] = None,
                   show: bool = False,
                   **kwargs):
        r""" visualize images of all fields
        
        Args:
            quantities: Determines what exactly is plotted. See
                :class:`~pde.visualization.plotting.ScalarfieldPlot` for
                details.
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Additional keyword arguments are passed to the plot
                methods of the individual fields
        """
        from ..visualization.plotting import ScalarFieldPlot
        plot = ScalarFieldPlot(self, quantities=quantities, show=show)
        if title is None:
            title = self.label
        plot.show_data(self, title=title)
        
        
    def plot(self, kind: str = 'auto', filename=None, **kwargs):
        r""" visualize the field
        
        Args:
            kind (str): Determines the visualizations. Supported values are
                `image`,  `line`, or `vector`. Alternatively, `auto` determines
                the best visualization based on the field itself.
            filename (str, optional): If given, the plot is written to the
                specified file. Otherwise, the plot might show directly in an
                interactive matplotlib session or `matplotlib.pyplot.show()`
                might be used to display the graphics.
            \**kwargs: All additional keyword arguments are forwarded to the
                actual plotting functions.
        """
        if kind == 'auto':
            self.plot_collection(**kwargs)
        elif kind == 'image':
            self.plot_image(**kwargs)
        elif kind == 'line':
            self.plot_line(**kwargs)
        elif kind == 'vector':
            self.plot_vector(**kwargs)
        else:
            raise ValueError(f'Unsupported plot `{kind}`. Possible choices are '
                             '`image`, `line`, `vector`, or `auto`.')
            
        # store the result to a file if requested
        if filename:
            import matplotlib.pyplot as plt
            plt.savefig(filename)
    
