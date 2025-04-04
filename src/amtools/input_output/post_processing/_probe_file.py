"""
_probe_file - defines the ProbeFile class.
"""


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

        This method performs the following:
        - Extracts probe indices and (x, y, z) coordinates from header comments.
        - Detects whether the file contains scalar or vector/tensor data.
        - Loads numerical time-series data efficiently using NumPy.
        - Assigns parsed data to `self.time`, `self.data`, and coordinate arrays.

        If the data is vector/tensor type, values are converted from string to numeric format
        and reshaped into a consistent 2D array format (n_timesteps, n_probes * vector_size).

        Raises:
            ValueError: If file format is invalid or cannot be parsed.
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
                    delimiter = r"\s+"
                    file_type = "scalar"
                else:
                    delimiter = r" {16}"
                    file_type = "vector/tensor"
                break  # No need to process further lines

        self.x = np.array(x_coords, dtype=np.float64)
        self.y = np.array(y_coords, dtype=np.float64)
        self.z = np.array(z_coords, dtype=np.float64)

        # Load numeric data
        data = np.loadtxt(self.path, comments="#", delimiter=None if delimiter == r"\s+" else 16)

        self.time = data[:, 0]
        self.data = data[:, 1:]

        if file_type == "vector/tensor":
            self.data = np.array([
                np.fromstring(x.strip("()"), sep=" ") for x in self.data.flatten()
            ]).reshape(data.shape[0], -1)
