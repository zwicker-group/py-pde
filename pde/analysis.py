'''
Functions for analyzing scalar fields

.. autosummary::
   :nosignatures:

   get_structure_factor
   get_length_scale

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import logging
import warnings
from functools import reduce

import numpy as np
from scipy import optimize

from .fields import ScalarField
from .grids.cartesian import CartesianGridBase



def get_structure_factor(scalar_field: ScalarField, ret_ks: bool = True):
    """ Calculates the structure factor associated with a scalar field
    
    Args:
        scalar_field (:class:`~pde.fields.ScalarField`):
            The scalar_field being analyzed
        ret_ks (bool):
            Flag determining whether the wave numbers associated with positions
            at which the structure factor is evaluated is returned
            
    Returns:
        (:class:`numpy.ndarray`, :class:`numpy.ndarray`):
            Two arrays giving the wave numbers and the associated structure
            factor. If `ret_ks` is False, only the second array is returned
    """
    if not isinstance(scalar_field, ScalarField):
        raise TypeError('Length scales can only be calculated for scalar '
                        f'fields, not {scalar_field.__class__.__name__}')
    
    grid = scalar_field.grid
    if not isinstance(grid, CartesianGridBase):
        raise NotImplementedError('Structure factor can currently only be '
                                  'calculated for Cartesian grids')
    if not all(grid.periodic):
        logger = logging.getLogger(__name__)
        logger.warning('Structure factor calculation assumes periodic boundary '
                       'conditions, but not all grid dimensions are periodic')
        
    # do the n-dimensional Fourier transform and calculate the absolute value
    sf = np.absolute(np.fft.fftn(scalar_field.data)).flat[1:]
    
    if ret_ks:
        # determine the (squared) components of the wave vectors
        k2s = [np.fft.fftfreq(grid.shape[i], d=grid.discretization[i])**2
               for i in range(grid.dim)]
        # calculate the magnitude 
        k_mag = np.sqrt(reduce(np.add.outer, k2s)).flat[1:]
        return k_mag, sf
    
    else:  # only return the structure factor
        return sf



def get_length_scale(scalar_field: ScalarField,
                     method: str = 'structure_factor_mean',
                     full_output: bool = False,
                     smoothing: float = None):
    """ Calculates a length scale associated with a scalar field
    
    Args:
        scalar_field (:class:`~pde.fields.ScalarField`):
            The scalar_field being analyzed
        method (str):
            A string determining which method is used to calculate the length
            scale.Valid options are `structure_factor_maximum` (numerically 
            determine the maximum in the structure factor) and
            `structure_factor_mean` (calculate the mean of the structure
            factor).
        full_output (bool):
            Flag determining whether additional data is returned. The format of
            the returned data depends on the method.
        smoothing (float, optional):
            Length scale that determines the smoothing of the radially averaged
            structure factor. If `None` it is automatically determined from the
            typical discretization of the underlying grid. This parameter is
            only used if `method = 'structure_factor_maximum'`
            
    Returns:
        float: The determine length scale
        
        If `full_output = True`, a tuple with the length scale and an additional
        object with further information is returned. 
    """
    from .tools.math import SmoothData1D
    logger = logging.getLogger(__name__)
    
    if (method == 'structure_factor_mean' or
            method == 'structure_factor_average'):
        # calculate the structure factor 
        k_mag, sf = get_structure_factor(scalar_field)
        length_scale = np.sum(sf) / np.sum(k_mag * sf)
        
        if full_output:
            return length_scale, sf
    
    elif (method == 'structure_factor_maximum' or 
            method == 'structure_factor_peak'):
        # calculate the structure factor 
        k_mag, sf = get_structure_factor(scalar_field)

        # smooth the structure factor
        if smoothing is None:
            grid = scalar_field.grid
            smoothing = 0.01 * grid.typical_discretization
        sf_smooth = SmoothData1D(k_mag, sf, sigma=smoothing)
        
        # find the maximum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            max_est = k_mag[np.argmax(sf)]
            bracket = np.array([0.2, 1, 5]) * max_est
            logger.debug("Search maximum of structure factor in interval "
                         f"{bracket}")
            try:
                result = optimize.minimize_scalar(lambda x: -sf_smooth(x),
                                                  bracket=bracket)
            except Exception:
                logger.exception('Could not determine maximal structure factor')
                length_scale = np.nan
            else:
                if not result.success:
                    logger.warning('Maximization of structure factor resulted '
                                   'in the following message: '
                                   f'{result.message}')
                length_scale = 1 / result.x
    
        if full_output:
            return length_scale, sf_smooth
        
    else:
        raise ValueError(f'Method {method} is not defined. Valid values are '
                         '`structure_factor_mean` and '
                         '`structure_factor_maximum`')
    
    # return only the length scale with out any additional information
    return length_scale
    
    
__all__ = ['get_structure_factor', 'get_length_scale']
    