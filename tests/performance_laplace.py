#!/usr/bin/env python3

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE_PATH))

import numpy as np
import numba

from pde.grids import UnitGrid, CylindricalGrid, SphericalGrid
from pde.grids.operators import cartesian, cylindrical, spherical
from pde.grids.operators.common import PARALLELIZATION_THRESHOLD_2D
from pde.grids.boundaries import Boundaries
from pde.tools.numba import jit, jit_allocate_out
from pde.tools.misc import estimate_computation_speed



def custom_laplace_2d_periodic(shape, dx=1):
    """ make laplace operator with periodic boundary conditions """
    dx_2 = 1 / dx**2  
    dim_x, dim_y = shape    
    parallel = (dim_x * dim_y >= PARALLELIZATION_THRESHOLD_2D**2)
    
    @jit_allocate_out(parallel=parallel)
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in numba.prange(dim_x):
            im = dim_x - 1 if i == 0 else i - 1
            ip = 0 if i == dim_x - 1 else i + 1
                
            j = 0
            jm = dim_y - 1
            jp = j + 1
            out[i, j] = (arr[i, jm] + arr[i, jp] + arr[im, j] + arr[ip, j] -
                         4 * arr[i, j]) * dx_2
                         
            for j in range(1, dim_y - 1):
                jm = j - 1
                jp = j + 1
                out[i, j] = (arr[i, jm] + arr[i, jp] + arr[im, j] + arr[ip, j] -
                             4 * arr[i, j]) * dx_2
                             
            j = dim_y - 1
            jm = j - 1
            jp = 0
            out[i, j] = (arr[i, jm] + arr[i, jp] + arr[im, j] + arr[ip, j] -
                         4 * arr[i, j]) * dx_2
        return out
    
    return laplace



def custom_laplace_2d_no_flux(shape, dx=1):
    """ make laplace operator with no-flux boundary conditions """
    dx_2 = 1 / dx**2  
    dim_x, dim_y = shape
    parallel = (dim_x * dim_y >= PARALLELIZATION_THRESHOLD_2D**2)

    @jit_allocate_out(parallel=parallel)
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in numba.prange(dim_x):
            im = 0 if i == 0 else i - 1
            ip = dim_x - 1 if i == dim_x - 1 else i + 1
                
            for j in range(dim_y):
                jm = 0 if j == 0 else j - 1
                jp = dim_y - 1 if j == dim_y - 1 else j + 1
                    
                out[i, j] = (arr[i, jm] + arr[i, jp] +
                             arr[im, j] + arr[ip, j] -
                             4 * arr[i, j]) * dx_2
        return out
    return laplace



def custom_laplace_2d(shape, periodic, dx=1):
    """ make laplace operator with no-flux or periodic boundary conditions """
    if periodic:
        return custom_laplace_2d_periodic(shape, dx=dx)
    else:
        return custom_laplace_2d_no_flux(shape, dx=dx)



def flexible_laplace_2d(bcs):
    """ make laplace operator with flexible boundary conditions """
    bc_x, bc_y = bcs
    dx = bcs._uniform_discretization
    dx_2 = 1 / dx**2
    dim_x, dim_y = bcs.grid.shape
     
    region_x = bc_x.make_region_evaluator()
    region_y = bc_y.make_region_evaluator()
     
    @jit_allocate_out
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in range(dim_x):
            for j in range(dim_y):
                val_x_l, val_x, val_x_r = region_x(arr, (i, j))
                val_y_l, _, val_y_r = region_y(arr, (i, j))

                out[i, j] = (val_x_l + val_x_r + val_y_l + val_y_r -
                             4 * val_x) * dx_2
        return out        
     
    return laplace



def custom_laplace_cyl_no_flux(shape, dr=1, dz=1):
    """ make laplace operator with no-flux boundary conditions """
    dim_r, dim_z = shape
    dr_2 = 1 / dr**2
    dz_2 = 1 / dz**2

    @jit
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        if out is None:
            out = np.empty((dim_r, dim_z))
            
        for j in range(0, dim_z):  # iterate axial points
            jm = 0 if j == 0 else j - 1
            jp = dim_z - 1 if j == dim_z - 1 else j + 1
                
            # inner radial boundary condition
            i = 0
            out[i, j] = (
                2 * (arr[i + 1, j] - arr[i, j]) * dr_2 +
                (arr[i, jm] + arr[i, jp] - 2 * arr[i, j]) * dz_2
            )
            
            for i in range(1, dim_r - 1):  # iterate radial points
                out[i, j] = (
                    (arr[i + 1, j] - 2 * arr[i, j] + arr[i - 1, j]) * dr_2 +
                    (arr[i + 1, j] - arr[i - 1, j]) / (2 * i + 1) * dr_2 +
                    (arr[i, jm] + arr[i, jp] - 2 * arr[i, j]) * dz_2
                )
                
            # outer radial boundary condition
            i = dim_r - 1
            out[i, j] = (
                (arr[i - 1, j] - arr[i, j]) * dr_2 +
                (arr[i, j] - arr[i - 1, j]) / (2 * i + 1) * dr_2 +
                (arr[i, jm] + arr[i, jp] - 2 * arr[i, j]) * dz_2
            )
        return out
        
    return laplace



def main():
    """ main routine testing the performance """
    print('Reports calls-per-second (larger is better)')
    print('  The `CUSTOM` method implemented by hand is the baseline case.')
    print('  The `FLEXIBLE` method is a serial implementation using the '
          'boundary conditions from the package.')
    print('  The other methods use the functions supplied by the package.\n')
    
    # Cartesian grid with different shapes and boundary conditions
    for shape in [(32, 32), (512, 512)]:
        data = np.random.random(shape)
        for periodic in [True, False]:
            grid = UnitGrid(shape, periodic=periodic)
            bcs = Boundaries.from_data(grid, 'natural')
            print(grid)
            result = cartesian.make_laplace(bcs, method='scipy')(data)
             
            for method in ['FLEXIBLE', 'CUSTOM', 'numba', 'matrix', 'scipy']:
                if method == 'FLEXIBLE':
                    laplace = flexible_laplace_2d(bcs)
                elif method == 'CUSTOM':
                    laplace = custom_laplace_2d(shape, periodic=periodic)
                elif method in {'numba', 'matrix', 'scipy'}:
                    laplace = cartesian.make_laplace(bcs, method=method)
                else:
                    raise ValueError(f'Unknown method `{method}`')
                # call once to pre-compile and test result 
                np.testing.assert_allclose(laplace(data), result)
                speed = estimate_computation_speed(laplace, data)
                print(f'{method:>8s}: {int(speed):>9d}')
            print()
 
    # Cylindrical grid with different shapes
    for shape in [(32, 64), (512, 512)]:
        data = np.random.random(shape)
        grid = CylindricalGrid(shape[0], [0, shape[1]], shape)
        print(f'Cylindrical grid, shape={shape}')
        bcs = Boundaries.from_data(grid, 'no-flux')
        laplace_cyl = cylindrical.make_laplace(bcs)
        result = laplace_cyl(data)
         
        for method in ['CUSTOM', 'numba']:
            if method == 'CUSTOM':
                laplace = custom_laplace_cyl_no_flux(shape)
            elif method == 'numba':
                laplace = laplace_cyl
            else:
                raise ValueError(f'Unknown method `{method}`')
            # call once to pre-compile and test result 
            np.testing.assert_allclose(laplace(data), result)
            speed = estimate_computation_speed(laplace, data)
            print(f'{method:>8s}: {int(speed):>9d}')
        print()

    # Spherical grid with different shapes
    for shape in [32, 512]:
        data = np.random.random(shape)
        grid = SphericalGrid(shape, shape)
        print(grid)
        bcs = Boundaries.from_data(grid, 'no-flux')        
        make_laplace = spherical.make_laplace 
        
        for conservative in [True, False]:
            laplace = make_laplace(bcs, conservative=conservative)
            # call once to pre-compile 
            laplace(data)
            speed = estimate_computation_speed(laplace, data)
            print(f' numba (conservative={str(conservative):<5}): '
                  f'{int(speed):>9d}')
        print()



if __name__ == '__main__':
    main()
