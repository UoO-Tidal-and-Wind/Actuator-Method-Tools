"""
Actuator Line Module

This module provides functions for creating and saving visualizations of actuator lines for turbine
blades and rotor geometries. It includes utilities for saving `.vtm` files for each timestep and
generating `.pvd` files that describe collections of these datasets over time. The primary functions
include generating actuator point visualizations along turbine blades and rotor geometry features
like rotor apex, base location, and tower shaft intersections.

Key Functions:
- `write_pvd`: Writes a `.pvd` file describing a collection of VTK datasets for each timestep.
- `create_blade_point_vtps`: Creates and saves `.vtm` files representing actuator points along a
turbine blade.
- `create_rotor_geometry_vtps`: Creates and saves `.vtm` files representing features of the rotor
geometry.

Imports:
    - pyvista (pv): Used for 3D geometry and visualization of actuator lines and rotor features.
    - numpy: Used for numerical operations, particularly array and matrix manipulations.
    - pathlib: Provides utilities for file path handling.
    - CaseReader (from input_output): Handles reading case data.
    - AnimationControl (from animation_control): Manages animation settings and time indices.
    - BladeData, TurbineLocationData (from turbine): Defines the data structures for turbine blades
    and locations.

Functions:
    - `write_pvd(filenames, times, output_path)`
    - `create_blade_point_vtps(blade_data, animation_control, point_size_scalar=1/200)`
    - `create_rotor_geometry_vtps(turbine_data, animation_control, point_size_scalar=1/50)`
"""

import pathlib
import pyvista as pv
import numpy as np
from .animation_control import AnimationControl
from ..data_structures.turbine import BladeData, TurbineLocationData


def write_pvd(filenames, times, output_path):
    """
    Write a .pvd file describing a collection of VTK datasets for each timestep.

    This function generates a `.pvd` (VTK File) that references the `.vtm` files for each timestep.
    It is useful for combining time-series data into a single file for visualization in applications
    like ParaView.

    Args:
        filenames (List[str]): List of file paths to the `.vtm` files to include in the `.pvd` file.
        times (List[float]): List of timesteps corresponding to the data in the `.vtm` files.
        output_path (str): The path where the `.pvd` file will be saved.

    Returns:
        None

    Example:
        write_pvd(["frame1.vtm", "frame2.vtm"], [0.0, 1.0], "output/turbine.pvd")
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for filename, time in zip(filenames, times):
            f.write(
                f'    <DataSet timestep="{time}" group="" part="0" file="{filename}"/>\n'
            )
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
    print(f"Saved .pvd file to {output_path}")


def create_blade_point_vtps(
    blade_data: BladeData,
    animation_control: AnimationControl,
    point_size_scalar: float = 1 / 200,
):
    """
    Create and save `.vtm` files for actuator points along the turbine blade at each timestep.

    This function generates `.vtm` files for actuator points on the turbine blade, using the blade
    root and tip positions at each timestep. It stores the actuator points in 3D space as spheres
    and saves them to disk with file names indicating the timestep.

    Args:
        blade_data (BladeData): The data structure containing blade positions and other
        necessary data for calculating actuator points.
        animation_control (AnimationControl): The object that controls the time-stepping of the
        animation, including selecting which frames to generate.
        point_size_scalar (float, optional): Scalar value to adjust the size of the points in the
        visualization. Defaults to 1/200.

    Returns:
        None

    Example:
        create_blade_point_vtps(blade_data, animation_control)
    """

    def solve_quadratics(a, b, c):
        discriminant = b**2 - 4 * a * c
        sqrt_discriminant = np.sqrt(discriminant.astype(np.complex128))
        x1 = (-b + sqrt_discriminant) / (2 * a)
        x2 = (-b - sqrt_discriminant) / (2 * a)
        return x1, x2

    output_path = animation_control.out_path

    time_arr = blade_data.time
    blade_station_radius_arr = blade_data.blade_station_radius

    rotor_radius = np.max(blade_station_radius_arr)
    point_size = point_size_scalar * rotor_radius

    frame_time_inds = animation_control.get_frame_time_indicies(t_arr=time_arr)

    for blade_ind, _ in enumerate(blade_station_radius_arr):
        file_names = []
        for time_ind in frame_time_inds:
            blade_root_pos = blade_data.blade_root_positions[blade_ind, time_ind]
            blade_tip_pos = blade_data.blade_tip_positions[blade_ind, time_ind]

            x_0 = blade_root_pos[0]
            y_0 = blade_root_pos[1]
            z_0 = blade_root_pos[2]

            x_1 = blade_tip_pos[0]
            y_1 = blade_tip_pos[1]
            z_1 = blade_tip_pos[2]

            # parameterised by some s in [0,1]
            x_grad = (x_1 - x_0) / (1 - 0)
            y_grad = (y_1 - y_0) / (1 - 0)
            z_grad = (z_1 - z_0) / (1 - 0)

            # then solve quadrastic system
            a = x_grad**2 + y_grad**2 + z_grad**2
            b = 2 * (x_grad * x_0 + y_grad * y_0 + z_grad * z_0)
            c = (
                blade_station_radius_arr[blade_ind, 0] ** 2
                - blade_station_radius_arr[blade_ind, :] ** 2
            )

            # solve quadratic equation for s(r)
            s1, s2 = solve_quadratics(a, b, c)

            # take the real component (should have zero imaginary component)
            s1 = np.real(s1)
            s2 = np.real(s2)

            # take the positive root
            s = np.maximum(s1, s2)

            x_points = x_grad * s + x_0
            y_points = y_grad * s + y_0
            z_points = z_grad * s + z_0

            points = np.column_stack((x_points, y_points, z_points))

            multiblock = pv.MultiBlock()
            for point in points:
                sphere = pv.Sphere(radius=point_size, center=point)
                multiblock.append(sphere)

            file_name = pathlib.Path(output_path).joinpath(
                f"blade{blade_ind}_t_{time_ind}.vtm"
            )

            file_names.append(file_name)
            print(f"saving: {file_name}")
            multiblock.save(file_name)

        pvd_path = pathlib.Path(output_path).joinpath(f"blade{blade_ind}.pvd")
        write_pvd(filenames=file_names, times=time_arr[frame_time_inds], output_path=pvd_path)


def create_rotor_geometry_vtps(
    turbine_data: TurbineLocationData,
    animation_control: AnimationControl,
    point_size_scalar: float = 1 / 50,
):
    """
    Create and save `.vtm` files for features of the rotor geometry at each timestep.

    This function generates `.vtm` files for key features of the rotor, such as the rotor apex,
    base location, and tower shaft intersection. These points are visualized as spheres at each
    timestep and saved to disk.

    Args:
        turbine_data (TurbineLocationData): The data structure containing turbine rotor data,
            including rotor apex, base location, and tower shaft intersection positions.
        animation_control (AnimationControl): The object that controls the time-stepping of
            the animation, including selecting which frames to generate.
        point_size_scalar (float, optional): Scalar value to adjust the size of the points in the
            visualization. Defaults to 1/50.

    Returns:
        None

    Example:
        create_rotor_geometry_vtps(turbine_data, animation_control)
    """
    output_path = animation_control.out_path

    time_arr = turbine_data.time

    rotor_radius = turbine_data.rotor_radius
    point_size = point_size_scalar * rotor_radius

    frame_time_inds = animation_control.get_frame_time_indicies(t_arr=time_arr)

    rotor_apex_dict = {"name": "rotorApex", "data": turbine_data.rotor_apex_position}
    base_location_dict = {
        "name": "baseLocation",
        "data": turbine_data.base_location_position,
    }
    tower_shaft_intersect_dict = {
        "name": "towerShaftIntersect",
        "data": turbine_data.tower_shaft_intersect_postiion,
    }

    features = [rotor_apex_dict, base_location_dict, tower_shaft_intersect_dict]

    for feature in features:
        name = feature["name"]
        data = feature["data"]
        file_names = []
        for time_ind in frame_time_inds:
            multiblock = pv.MultiBlock()
            multiblock.append(pv.Sphere(radius=point_size, center=data[time_ind]))

            file_name = pathlib.Path(output_path).joinpath(f"{name}_t_{time_ind}.vtm")
            file_names.append(file_name)
            print(f"saving: {file_name}")
            multiblock.save(file_name)

        pvd_path = pathlib.Path(output_path).joinpath(f"{name}.pvd")
        write_pvd(filenames=file_names, times=time_arr[frame_time_inds], output_path=pvd_path)
