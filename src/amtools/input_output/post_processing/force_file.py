"""
Reader for OpenFOAM ``force.dat`` post-processing output.
"""

import logging
from pathlib import Path
import numpy as np

class ForceFile:
    """
    Reader for force post-processing data.

    Attributes:
        path: Path to ``force.dat``.
        time: One-dimensional array of output times.
        data: Numeric force columns after the time column.
        total: First three data columns after :meth:`split_data`.
        pressure: Placeholder attribute initialized for pressure-force data.
        viscous: Columns six to eight after :meth:`split_data`.
    """
    def __init__(self, file_path: str):
        """
        Initialize a force-file reader and validate the path.

        Args:
            file_path: Path to the ``force.dat`` file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        self.time = np.array([])
        self.data = np.array([])

        self.total = np.array([])
        self.pressure = np.array([])
        self.ptressure = self.pressure
        self.viscous = np.array([])

    def set_path(self, file_path: str):
        """
        Update the path to the force output file.

        Args:
            file_path: New path to ``force.dat``.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def read(self):
        """
        Read numeric force data from disk.

        The first column is stored as ``time`` and all remaining columns are stored as
        ``data`` before :meth:`split_data` populates convenience slices.
        """

        # assumes order of data is total, pressure, viscous
        data = np.genfromtxt(self.path, comments="#")
        self.time = data[:,0]
        self.data = data[:,1:]
        self.split_data()

    def split_data(self):
        """
        Split loaded force columns into convenience arrays.

        The current implementation assumes the non-time columns are ordered as total,
        pressure, and viscous force components with three components per group.
        """
        self.total = self.data[:, 0:3]
        self.pressure = self.data[:, 3:6]
        self.ptressure = self.pressure
        self.viscous = self.data[:, 6:9]

    def crop_by_time(self, lower_limit: float = None, upper_limit: float = None):
        """
        Crop force data to a closed time interval in-place.

        Rows are retained when they satisfy the provided lower and upper bounds.

        Args:
            lower_limit: Optional lower time bound.
            upper_limit: Optional upper time bound.
        """
        time_mask = np.ones_like(self.time, dtype=bool)

        if lower_limit is not None:
            time_mask &= self.time >= lower_limit
        if upper_limit is not None:
            time_mask &= self.time <= upper_limit

        self.time = self.time[time_mask]
        self.data = self.data[time_mask, :]
        self.split_data()
