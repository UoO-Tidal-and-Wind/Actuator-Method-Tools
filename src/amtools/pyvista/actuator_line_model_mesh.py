"""
actuator_line_model_mesh.py
===============

This module provides utilities for visualising the actuator line model in paraview.
"""

import pyvista as pv
import numpy as np
from ..input_output import CaseReader, BladeData


def create_blade_vtps(blade_data: BladeData) -> list:
    
    def solve_quadratics(a, b, c):
        discriminant = b**2 - 4*a*c
        sqrt_discriminant = np.sqrt(discriminant.astype(np.complex128))
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        return x1, x2

    time_arr = blade_data.time
    num_of_blades = blade_data.number_of_blades
    blade_station_radius_arr = blade_data.blade_station_radius

    for time_ind, time in enumerate(time_arr):
        multiblock = pv.MultiBlock()
        for blade_ind, blade_station_arr in enumerate(blade_station_radius_arr):
            blade_root_pos = blade_data.blade_root_positions[blade_ind, time_ind]
            blade_tip_pos = blade_data.blade_root_positions[blade_ind, time_ind]
            
            x_0 = blade_root_pos[0]
            y_0 = blade_root_pos[1]
            z_0 = blade_root_pos[2]
            
            x_1 = blade_tip_pos[0]
            y_1 = blade_tip_pos[1]
            z_1 = blade_tip_pos[2]
            
            # parameterised by some s in [0,1]
            x_grad = (x_1-x_0)/(1-0)
            y_grad = (y_1-y_0)/(1-0)
            z_grad = (z_1-z_0)/(1-0)
            
            # then solve quadrastic system
            a = (x_grad**2 + y_grad**2 +z_grad**2)
            b = 2*(x_grad*x_0 + y_grad*y_0 + z_grad*z_0)
            c = (blade_station_radius_arr**2 - blade_station_radius_arr[blade_ind,0]**2)
            
            x1, x2 = solve_quadratics(a,b,c)
            
            
            
            
