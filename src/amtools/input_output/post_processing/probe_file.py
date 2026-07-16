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

from __future__ import annotations
import gzip
from typing import Union, Sequence
import logging
from pathlib import Path
import re
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        Initializes an empty ProbeFile instance.

        Args:
            file_path (str): Path to the probe output file.

        The path may point to a file that does not yet exist, allowing a probe
        file to be populated and written with the write method.
        """
        self.path = file_path

        self.time = np.array([])
        self.data = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        
        self.include_out_bounds = True

    def set_path(self, file_path: str):
        """
        Updates the file path to a new probe output file.

        Args:
            file_path (str): Path to the new probe file.

        The path may point to a new output file.
        """
        self.path = file_path

    def read(self):
        """
        Reads and parses the probe file.
        """
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")
        #probe_pattern = re.compile(
        #    r"# Probe (\d+) \((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)"
        #)

        float_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        probe_pattern = re.compile(
            rf"# Probe (\d+) \(({float_pattern}) ({float_pattern}) ({float_pattern})\)"
        )


        probe_indices = []
        x_coords = []
        y_coords = []
        z_coords = []

        # Read file efficiently
        open_file = gzip.open if Path(self.path).suffix == ".gz" else open
        with open_file(self.path, "rt", encoding="utf-8") as file:
            lines = file.readlines()

        # Find delimiter and file type in a single pass
        # delimiter = None
        file_type = None
        for line in lines:
            if line.startswith("#"):
                if "Time" in line:
                    continue
                if "# Not Found" in line:
                    if not (self.include_out_bounds):
                        continue
                match = probe_pattern.match(line)
                if match:
                    probe_indices.append(int(match.group(1)))
                    x_coords.append(float(match.group(2)))
                    y_coords.append(float(match.group(3)))
                    z_coords.append(float(match.group(4)))
            else:
                if "(" not in line:
                    # delimiter = r"\s+"  # Handle spaces or tabs
                    file_type = "scalar"
                else:
                    # delimiter = r" {16}"  # Expect exactly 16 spaces
                    file_type = "vector/tensor"
                break  # No need to process further lines

        self.x = np.array(x_coords, dtype=np.float64)
        self.y = np.array(y_coords, dtype=np.float64)
        self.z = np.array(z_coords, dtype=np.float64)
        
        pattern = re.compile(
            r"""
            ^\s*
            ([+-]?\d+(?:\.\d+)?)      # time (float)
            \s+
            (.*)$                     # rest of the line
            """,
            re.VERBOSE,
        )

        data = np.array([])
        # Use regular expressions to split the data lines
        if file_type == "scalar":
            data = np.loadtxt(lines, comments="#")
        elif file_type == "vector/tensor":
            # Split by 16 spaces and load the data manually
            with open_file(self.path, "rt", encoding="utf-8") as file:
                # Skip header lines
                data_lines = [line for line in file if not line.startswith("#")]

            # Apply regex to split by 16 spaces for each line
            # data = np.array([re.split(r" {16}", line.strip()) for line in data_lines])
            data = np.array([
                [m.group(1), *re.findall(r"\([^()]*\)", m.group(2))]
                for line in data_lines
                if (m := pattern.search(line))
            ], dtype=object)
        else:
            raise Exception("Could not infer file_type")
        
        self.time = np.array(data[:, 0], dtype=float)
        self.data = data[:, 1:]

        if file_type == "vector/tensor":
            # Initialize the data structure as a list of lists (2D structure)
            self.data = np.array(
                [
                    [np.array(x.strip("()").split(" "), dtype=float) for x in row]
                    for row in self.data
                ]
            )

    def get_probe_data_by_probe_index(
        self, probe_index: Union[int, Sequence[int]]
    ) -> np.ndarray:
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

    def crop_by_time(self, lower_limit: float = -1e10, upper_limit: float = 1e10):
        """
        Crops probe data and time array to be within the lower and upper limits.

        Args:
            lower_limit (float, optional): Lower limit for cropping. Defaults to -1E10.
            upper_limit (float, optional): Upper limit for cropping. Defaults to 1E10.
        """
        mask = (self.time > lower_limit) & (self.time < upper_limit)

        self.data = self.data[mask]
        self.time = self.time[mask]

    def get_data_by_component(
        self, component_index: Union[int, Sequence[int]]
    ) -> np.ndarray:
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
        
        
    def find_nearest_probe(self, x: float, y: float, z: float) -> int:
        """
        Returns probe index of probe nearest to the specified coordinates.

        Args:
            x (float): X-coordinate of the target location.
            y (float): Y-coordinate of the target location.
            z (float): Z-coordinate of the target location.
        """
        
        distances = np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2)
        return int(np.argmin(distances))
    
    def append(self, other: ProbeFile):
        if self.time.max() != other.time.max():
            print("Times do not match")
            raise ValueError
        
        self.x = np.append(self.x, other.x)
        self.y = np.append(self.y, other.y)
        self.z = np.append(self.z, other.z)
        self.data = np.concatenate((self.data, other.data), axis=1)


    @staticmethod
    def _format_number(value) -> str:
        """Format a numeric value without sacrificing floating-point precision."""
        return repr(float(value))

    def _probe_header_label(self) -> str:
        return "Probe"

    def _row_prefix(self, row_index: int) -> list[str]:
        return [self._format_number(self.time[row_index])]

    def _validate_for_write(self) -> None:
        """Validate the in-memory arrays before serialising a probe file."""
        coordinates = (np.asarray(self.x), np.asarray(self.y), np.asarray(self.z))
        if any(coordinate.ndim != 1 for coordinate in coordinates):
            raise ValueError("Probe coordinates must be one-dimensional arrays.")
        if len({len(coordinate) for coordinate in coordinates}) != 1:
            raise ValueError("x, y, and z must contain the same number of probes.")

        self.time = np.asarray(self.time)
        self.data = np.asarray(self.data)
        if self.time.ndim != 1 or len(self.time) == 0:
            raise ValueError("time must be a non-empty one-dimensional array.")
        if self.data.ndim not in (2, 3):
            raise ValueError(
                "data must have shape (time, probes) or (time, probes, components)."
            )
        if self.data.shape[0] != len(self.time):
            raise ValueError("data and time must have the same number of rows.")
        if self.data.shape[1] != len(self.x):
            raise ValueError("data and coordinates must have the same number of probes.")
        if self.data.shape[1] == 0 or (self.data.ndim == 3 and self.data.shape[2] == 0):
            raise ValueError("data must contain at least one probe value.")

    def write(self, file_path: str | Path | None = None) -> None:
        """Write the in-memory probe data in the format accepted by read."""
        if file_path is not None:
            self.path = file_path
        self._validate_for_write()

        target_path = Path(self.path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        open_file = gzip.open if target_path.suffix == ".gz" else open
        with open_file(target_path, "wt", encoding="utf-8") as file:
            for index, (x_value, y_value, z_value) in enumerate(zip(self.x, self.y, self.z)):
                coordinates = " ".join(
                    self._format_number(value) for value in (x_value, y_value, z_value)
                )
                file.write(f"# {self._probe_header_label()} {index} ({coordinates})\n")
            file.write("# Time\n")

            for row_index in range(len(self.time)):
                values = self.data[row_index]
                row = self._row_prefix(row_index)
                if self.data.ndim == 2:
                    row.extend(self._format_number(value) for value in values)
                else:
                    row.extend(
                        "(" + " ".join(self._format_number(value) for value in probe) + ")"
                        for probe in values
                    )
                file.write(" ".join(row) + "\n")
