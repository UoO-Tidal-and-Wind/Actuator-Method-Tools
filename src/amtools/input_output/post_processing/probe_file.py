"""
_probe_file
===========

Defines the `ProbeFile` class.

This module provides a class for representing and processing OpenFOAM-style probe 
data files. The class can load a probe output file, parse probe coordinates, 
read scalar or vector/tensor time-series data, and store the data for further analysis.

Attributes:
    path (str): Path to the probe data file.
    time (np.ndarray): 1D array of time values from the file.
    data (np.ndarray): 2D or 3D array of measurement data depending on file type.
    x (np.ndarray): X-coordinates of probe locations.
    y (np.ndarray): Y-coordinates of probe locations.
    z (np.ndarray): Z-coordinates of probe locations.

Example:
    Example usage of the `ProbeFile` class:

    ```python
    probe_file = ProbeFile('probe_data.txt')
    probe_file.read()
    print(probe_file.time)
    ```
"""


from typing import Union, Sequence
import logging
from pathlib import Path
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ProbeFile:
    """
    A class to represent and process OpenFOAM-style probe data files.

    This class is designed to:
    - Load a probe output file
    - Parse probe coordinates
    - Read scalar or vector/tensor time-series data
    - Store time and data arrays for further analysis

    Attributes:
    -----------
    path : str
        Path to the probe data file.
    time : np.ndarray
        1D array of time values from the file.
    data : np.ndarray
        2D or 3D array of measurement data depending on file type.
    x : np.ndarray
        X-coordinates of probe locations.
    y : np.ndarray
        Y-coordinates of probe locations.
    z : np.ndarray
        Z-coordinates of probe locations.
    """

    def __init__(self, file_path: str):
        """
        Initializes the ProbeFile instance and verifies the file exists.

        Args:
            file_path (str): Path to the probe output file.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        self.time = np.array([])
        self.data = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])

    def set_path(self, file_path: str):
        """
        Updates the file path to a new probe output file.

        Args:
            file_path (str): Path to the new probe file.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def read(self):
        """
        Reads and parses the probe file.
        """
        probe_pattern = re.compile(
            r"# Probe (\d+) \((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)"
        )

        probe_indices = []
        x_coords = []
        y_coords = []
        z_coords = []

        # Read file efficiently
        with open(self.path, "r", encoding='utf-8') as file:
            lines = file.readlines()

        # Find delimiter and file type in a single pass
        delimiter = None
        file_type = None
        for line in lines:
            if line.startswith("#"):
                if "Time" in line or "# Not Found" in line:
                    continue
                match = probe_pattern.match(line)
                if match:
                    probe_indices.append(int(match.group(1)))
                    x_coords.append(float(match.group(2)))
                    y_coords.append(float(match.group(3)))
                    z_coords.append(float(match.group(4)))
            else:
                if "(" not in line:
                    delimiter = r"\s+"  # Handle spaces or tabs
                    file_type = "scalar"
                else:
                    delimiter = r" {16}"  # Expect exactly 16 spaces
                    file_type = "vector/tensor"
                break  # No need to process further lines

        self.x = np.array(x_coords, dtype=np.float64)
        self.y = np.array(y_coords, dtype=np.float64)
        self.z = np.array(z_coords, dtype=np.float64)

        data = np.array([])
        # Use regular expressions to split the data lines
        if file_type == "scalar":
            data = np.loadtxt(self.path, comments="#")
        elif file_type == "vector/tensor":
            # Split by 16 spaces and load the data manually
            with open(self.path, "r", encoding='utf-8') as file:
                # Skip header lines
                data_lines = [
                    line for line in file if not line.startswith("#")]

            # Apply regex to split by 16 spaces for each line
            data = np.array(
                [re.split(r" {16}", line.strip()) for line in data_lines])

        self.time = np.array(data[:, 0], dtype=float)
        self.data = data[:, 1:]

        if file_type == "vector/tensor":
            # Initialize the data structure as a list of lists (2D structure)
            self.data = np.array([
                [np.array(x.strip("()").split(" "), dtype=float)
                 for x in row]
                for row in self.data
            ])

    def get_probe_data_by_probe_index(self, probe_index: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Get the data for a specified probe or probes.

        Args:
            probe_index (int or list of int): Index or indices of the specified probe(s).

        Returns:
            np.ndarray: Probe data of the specified probe(s).
        """
        return self.data[:, probe_index]

    def crop_data_by_probe_index(self, probe_index: Union[int, Sequence[int]]):
        """
        Crops the data to contain only the specified probe index or indices.

        Args:
            probe_index (int or list of int): Index or indices of probe(s) to crop to.
        """
        if isinstance(probe_index, int):
            probe_index = [probe_index]

        self.data = self.data[:, probe_index]
        self.x = self.x[probe_index]
        self.y = self.y[probe_index]
        self.z = self.z[probe_index]

    def crop_by_time(self, lower_limit: float = -1E10, upper_limit: float = 1E10):
        """
        Crops probe data and time array to be within the lower and upper limits.

        Args:
            lower_limit (float, optional): Lower limit for cropping. Defaults to -1E10.
            upper_limit (float, optional): Upper limit for cropping. Defaults to 1E10.
        """
        mask = (self.time > lower_limit) & (self.time < upper_limit)

        self.data = self.data[mask]
        self.time = self.time[mask]

    def get_data_by_component(self, component_index: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Returns the data array with only the specified components

        Args:
            component_index (Union[int, Sequence[int]]): Component indicies to return.

        Returns:
            np.ndarray: Data array containing only the provided components.
        """
        return self.data[:, :, component_index]

    def crop_by_component_index(self, component_index: Union[int, Sequence[int]]):
        """
        Crops data array by component indicies.

        Args:
            component_index (Union[int, Sequence[int]]): Indicies to crop to.
        """
        if isinstance(component_index, int):
            component_index = [component_index]

        self.data = self.data[:, :, component_index]
