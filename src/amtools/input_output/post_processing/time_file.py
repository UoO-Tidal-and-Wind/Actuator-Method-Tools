"""
Reader for OpenFOAM ``timeInfo.dat`` post-processing output.
"""

import logging
from pathlib import Path
import numpy as np
import re


class TimeFile:
    """
    Reader for runtime timing information.

    Attributes:
        path: Path to ``timeInfo.dat``.
        time: One-dimensional array of simulation times.
        data: Numeric timing columns after the time column.
        cumulative_cpu_time: Cumulative CPU time from the first data column.
        cumulative_clock_time: Cumulative wall-clock time from the second data column.
        cpu_time_per_step: Per-step CPU time when present.
        clock_time_per_step: Per-step wall-clock time when present.
    """
    def __init__(self, file_path: str):
        """
        Initialize a time-info reader and validate the path.

        Args:
            file_path: Path to the ``timeInfo.dat`` file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        self.time = np.array([])
        self.data = np.array([])

        self.cumulative_cpu_time = np.array([])
        self.cumulative_clock_time = np.array([])
        self.cpu_time_per_step = np.array([])
        self.clock_time_per_step = np.array([])

    def set_path(self, file_path: str):
        """
        Update the path to the time-info output file.

        Args:
            file_path: New path to ``timeInfo.dat``.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def read(self):
        """
        Read numeric timing data from disk.

        The first column is stored as ``time`` and all remaining columns are stored as
        ``data`` before :meth:`split_data` populates convenience arrays.
        """

        # assumes order of data is total, pressure, viscous
        data = np.genfromtxt(self.path, comments="#")
        self.time = data[:, 0]
        self.data = data[:, 1:]
        self.split_data()

    def split_data(self):
        """
        Populate convenience arrays from the loaded timing columns.

        The first two data columns are treated as cumulative CPU and wall-clock time.
        When at least four data columns are present, the third and fourth columns are
        also exposed as per-step CPU and wall-clock time.
        """
        self.total = self.data[:, 0:3]
        self.ptressure = self.data[:, 3:6]
        self.viscous = self.data[:, 6:9]

        self.cumulative_cpu_time = self.data[:, 0]
        self.cumulative_clock_time = self.data[:, 1]
        if self.data.shape[1] > 2:
            self.cpu_time_per_step = self.data[:, 2]
            self.clock_time_per_step = self.data[:, 3]

    def crop_by_time(self, lower_limit: float = None, upper_limit: float = None):
        """
        Crop timing data to a closed time interval in-place.

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
