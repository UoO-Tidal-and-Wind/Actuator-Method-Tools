"""
case_reader
===========

This module defines the `CaseReader` class, which provides functionality for reading and 
handling case directories related to turbine output and post-processing results.

The `CaseReader` class facilitates the following:
- Accessing turbine output data.
- Retrieving post-processing probe data.
- Navigating through time directories based on specific criteria (latest, first, exactly, closest
    to).
  
Classes:
    CaseReader: Handles access to turbine output and post-processing data from a case directory.
    
Example:
    Example usage of the `CaseReader` class:

    ```python
    case = CaseReader("/path/to/case")
    turbine_data = case.turbine_output("turbine_data.dat", time_dir="latest")
    probe_data = case.probe("probe_1", "U", time_dir="first")
    ```
"""

import logging
from pathlib import Path
from typing import Literal
import numpy as np

from .turbine_output.turbine_output_file import TurbineOutputFile
from .post_processing.probe_file import ProbeFile

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BladeData:
    def __init__(self):
        self.number_of_blades = 0
        self.time = np.array([])
        self.blade_tip_positions = np.array([])
        self.blade_root_positions = np.array([])
        self.blade_station_chord = np.array([])
        self.blade_station_radius = np.array([])

    def read_data_from_case(self, case_path: str):
        case_reader = CaseReader(case_path)

        # first read in the data
        radius_data = case_reader.turbine_output("radiusC")
        chord_data = case_reader.turbine_output("chordC")
        blade_tip_data = case_reader.turbine_output("bladeTipPosition")
        blade_root_data = case_reader.turbine_output("bladeRootPosition")

        # now fill out the member variables
        # (for now assume there are three blades and the radius
        # file only contains information at time 0)
        self.number_of_blades = len(np.unique(radius_data.blade))
        self.blade_station_radius = radius_data.data
        self.blade_station_chord = chord_data.data
        self.time = np.unique(blade_tip_data.time)
        self.blade_tip_positions = np.array(
            [blade_tip_data.get_using_blade_index(i) for i in range(self.number_of_blades)])
        self.blade_root_positions = np.array(
            [blade_root_data.get_using_blade_index(i) for i in range(self.number_of_blades)])


class CaseReader:
    """
    Class used to read in data from a case directory.

    This class manages paths related to turbine output and post-processing 
    within the specified case directory.
    """

    def __init__(self, root_path: str) -> None:
        """
        Initializes the CaseReader with the specified case directory.

        Args:
            root_path (str): Path to the root directory of the case.

        Raises:
            FileNotFoundError: If the provided root path does not exist.
        """
        self.path: Path = Path(root_path)  # Convert to Path object
        if not self.path.exists():
            logging.warning("The '%s' directory is missing.", self.path)
            raise FileNotFoundError(
                f"The directory '{root_path}' does not exist.")

        self.name: str = self.path.name  # Direct access to the name of the path
        self.turbine_output_path: Path = self.path / "turbineOutput"
        self.post_processing_path: Path = self.path / "postProcessing"
        self.turbine_output_time_dir = ""

        if not self.turbine_output_path.exists():
            logging.warning(
                "The 'turbineOutput' directory is missing in '%s'.", self.path)

        if not self.post_processing_path.exists():
            logging.warning(
                "The 'postProcessing' directory is missing in '%s'.", self.path)

    def set_path(self, root_path: str):
        """
        Updates the path for the case directory.

        Args:
            root_path (str): The new path to the root directory of the case.
        """
        self.path = Path(root_path)

    def turbine_output(self,
                    file_name: str,
                    time_dir: Literal["latest", "first", "exactly", "closest to"] = "latest",
                    time_dir_value: str = ""):
        """
        Retrieves turbine output data from the case directory.

        Args:
            file_name (str): Name of the turbine output file.
            time_dir (Literal["latest", "first", "exactly", "closest to"], optional): 
                Determines which time directory to select.
                - "latest": Uses the latest available time.
                - "first": Uses the earliest available time.
                - "exactly": Uses the time specified in `time_dir_value`.
                - "closest to": Finds the closest available time to `time_dir_value`.
                Defaults to "latest".
            time_dir_value (str, optional): Required if `time_dir` is "exactly" or "closest to".
                Specifies the exact or target time value.

        Returns:
            TurbineOutputFile: An instance of the TurbineOutputFile class after reading the file.

        Raises:
            ValueError: If `time_dir` is "exactly" or "closest to" but `time_dir_value` 
            is not provided.
        """
        # Find all the possible time directories
        time_dirs = [f.name for f in self.turbine_output_path.iterdir()]
        if time_dir == "latest":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmax(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "first":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "exactly":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'exactly'.")
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'.")
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = self.turbine_output_path / self.turbine_output_time_dir / file_name

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(
                f"The file '{file_path}' does not exist.")


        file_reader = TurbineOutputFile(file_path)
        file_reader.read()
        return file_reader

    def probe(self,
            probe_name: str,
            variable_name: str,
            time_dir: Literal["latest", "first", "exactly", "closest to"] = "latest",
            time_dir_value: str = ""):
        """
        Locate and load a probe file for a given variable from a specified time directory.

        Parameters
        ----------
        probe_name : str
            Name of the probe directory within the post-processing output.
        variable_name : str
            Name of the variable to load (e.g., 'U', 'p').
        time_dir : Literal["latest", "first", "exactly", "closest to"], optional
            Strategy for selecting the time directory:
            - "latest": use the highest time value (default)
            - "first": use the lowest time value
            - "exactly": use the directory that exactly matches `time_dir_value`
            - "closest to": use the directory closest to `time_dir_value`
        time_dir_value : str, optional
            Required if `time_dir` is "exactly" or "closest to". Should be a string representation
            of a floating-point time (e.g., "12.5").

        Returns
        -------
        ProbeFile
            A `ProbeFile` object containing the data read from the selected file.

        Raises
        ------
        FileNotFoundError
            If the specified probe directory or variable file does not exist.
        ValueError
            If `time_dir_value` is not provided when required.
        """

        probe_path = self.post_processing_path / probe_name

        if not probe_path.exists():
            logging.warning("The '%s' path is missing.", probe_path)
            raise FileNotFoundError(
                f"The path '{probe_path}' does not exist.")

        time_dirs = [f.name for f in probe_path.iterdir()]
        if time_dir == "latest":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmax(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "first":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "exactly":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'exactly'.")
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'.")
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = probe_path / self.turbine_output_time_dir / variable_name

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(
                f"The file '{file_path}' does not exist.")

        file_reader = ProbeFile(file_path)
        file_reader.read()
        return file_reader
    
    def get_blade_data(self) -> BladeData:
        blade_data = BladeData()
        blade_data.read_data_from_case(self.path)

