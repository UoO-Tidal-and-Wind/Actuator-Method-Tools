"""
This module contains classes and methods for handling turbine blade data and turbine location data.

The `BladeData` class manages information about the turbine blades, including the number of blades,
their positions, and blade station parameters (chord and radius). It includes functionality to read
this data from a specified case file.

The `TurbineLocationData` class handles the location data of the turbine, including the rotor apex
position, tower shaft intersection position, base location, and rotor radius. It also includes
functionality for reading this data from a case file.

Both classes depend on the `CaseReader` class from the `input_output.case_reader` module to extract
data from the provided case files.

Classes:
    BladeData: A class for storing and managing turbine blade data.
    TurbineLocationData: A class for storing and managing turbine location data.

"""

import numpy as np
from ..input_output.case_reader import CaseReader


class BladeData:
    """
    A class to represent the blade data of a turbine.

    This class stores data related to the turbine blades, including the number of blades,
    blade tip positions, blade root positions, and the station radii and chords for each blade.
    It also provides a method to read the blade data from a specified case file.

    Attributes:
        number_of_blades (int): The number of blades in the turbine.
        time (np.ndarray): An array of time steps associated with the blade data.
        blade_tip_positions (np.ndarray): An array of positions for the tip of the blades.
        blade_root_positions (np.ndarray): An array of positions for the root of the blades.
        blade_station_chord (np.ndarray): An array of chord lengths for each blade station.
        blade_station_radius (np.ndarray): An array of radii for each blade station.

    Methods:
        read_data_from_case(case_path: str): Reads the blade data from the provided case file path.
    """

    def __init__(self):
        """
        Initializes the BladeData object with default values.
        """
        self.number_of_blades = 0
        self.time = np.array([])
        self.blade_tip_positions = np.array([])
        self.blade_root_positions = np.array([])
        self.blade_station_chord = np.array([])
        self.blade_station_radius = np.array([])

    def read_data_from_case(self, case_path: str):
        """
        Reads the blade data from the specified case file.

        This method uses the `CaseReader` to read the necessary data (radius, chord, blade tip,
        and blade root positions) from the case file and populates the corresponding attributes of
        the `BladeData` class.

        Args:
            case_path (str): The path to the case file containing the blade data.

        Returns:
            None

        Raises:
            ValueError: If the case file doesn't contain the expected data.
        """
        case_reader = CaseReader(case_path)

        # first read in the data
        radius_data = case_reader.turbine_output("radiusC")
        chord_data = case_reader.turbine_output("chordC")
        blade_tip_data = case_reader.turbine_output("bladeTipPosition")
        blade_root_data = case_reader.turbine_output("bladeRootPosition")

        # now fill out the member variables
        self.number_of_blades = len(np.unique(radius_data.blade))
        self.blade_station_radius = radius_data.data
        self.blade_station_chord = chord_data.data
        self.time = np.unique(blade_tip_data.time)
        self.blade_tip_positions = np.array(
            [
                blade_tip_data.get_using_blade_index(i)
                for i in range(self.number_of_blades)
            ]
        )
        self.blade_root_positions = np.array(
            [
                blade_root_data.get_using_blade_index(i)
                for i in range(self.number_of_blades)
            ]
        )


class TurbineLocationData:
    """
    A class to represent the location data of a turbine.

    This class stores the positions of various features of the turbine rotor, including
    the rotor apex, tower shaft intersection, base location, and rotor radius. It also provides
    a method to read the location data from a specified case file.

    Attributes:
        time (np.ndarray): An array of time steps associated with the location data.
        rotor_apex_position (np.ndarray): An array of positions for the rotor apex.
        tower_shaft_intersect_postiion (np.ndarray): An array of positions for the tower shaft
        intersection.
        base_location_position (np.ndarray): An array of positions for the base location.
        rotor_radius (float): The radius of the rotor.

    Methods:
        read_data_from_case(case_path: str): Reads the location data from the provided case file
        path.
    """

    def __init__(self):
        """
        Initializes the TurbineLocationData object with default values.
        """
        self.time = np.array([])
        self.rotor_apex_position = np.array([])
        self.tower_shaft_intersect_postiion = np.array([])
        self.base_location_position = np.array([])
        self.rotor_radius = 0

    def read_data_from_case(self, case_path: str):
        """
        Reads the location data from the specified case file.

        This method uses the `CaseReader` to read the necessary data (rotor apex, tower shaft
        intersection, base location, and rotor radius) from the case file and populates the
        corresponding attributes of the `TurbineLocationData` class.

        Args:
            case_path (str): The path to the case file containing the location data.

        Returns:
            None

        Raises:
            ValueError: If the case file doesn't contain the expected data.
        """
        case_reader = CaseReader(case_path)

        tower_shaft_intersect_data = case_reader.turbine_output(
            "towerShaftIntersectPosition"
        )
        rotor_apex_data = case_reader.turbine_output("rotorApexPosition")
        base_location_data = case_reader.turbine_output("baseLocationPosition")

        radius_data = case_reader.turbine_output("radiusC")

        self.rotor_apex_position = rotor_apex_data.data
        self.tower_shaft_intersect_postiion = tower_shaft_intersect_data.data
        self.base_location_position = base_location_data.data

        self.time = rotor_apex_data.time
        self.rotor_radius = np.max(radius_data.data)
