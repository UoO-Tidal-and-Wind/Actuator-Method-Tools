"""
Reader for OpenFOAM planar averaging output files.

Planar averaging files contain one height header, one column-description header,
and then rows of ``time``, ``dt``, and one averaged value for each height.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
import re

import numpy as np


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PlanarAverageFile:
    """
    Reader for planar averaging output.

    Attributes:
        path: Path to the planar averaging output file.
        height: One-dimensional array of averaging heights.
        time: One-dimensional array of output times.
        dt: One-dimensional array of timesteps used for each output.
        data: Averaged values. Scalars have shape ``(time, heights)``; vectors and
            tensors have shape ``(time, heights, components)``.
    """

    _FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    def __init__(self, file_path: str | Path):
        """
        Initialize an empty planar-average reader.

        Args:
            file_path: Path to the planar averaging output file.
        """
        self.path = file_path

        self.height = np.array([])
        self.time = np.array([])
        self.dt = np.array([])
        self.data = np.array([])

    def set_path(self, file_path: str | Path):
        """
        Update the path to the planar averaging output file.

        Args:
            file_path: New path to use for subsequent reads.
        """
        self.path = file_path

    def read(self):
        """
        Read heights, times, timesteps, and averaged values from disk.

        The first line is expected to contain heights, for example
        ``Height (m) 1.25 2.5 3.75``. The second line is treated as a column label row.
        Remaining rows contain ``time``, ``dt``, and one scalar or parenthesized
        vector/tensor value per height.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is incomplete or the parsed data shape is
                inconsistent with the height header.
        """
        path = Path(self.path)
        if not path.exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")

        open_file = gzip.open if path.suffix == ".gz" else open
        with open_file(path, "rt", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if len(lines) < 3:
            raise ValueError(
                f"The file '{self.path}' does not contain a complete planar average."
            )

        self.height = self._parse_height_line(lines[0])
        data_lines = [line for line in lines[2:] if not line.lstrip().startswith("#")]
        if not data_lines:
            raise ValueError(f"The file '{self.path}' does not contain data rows.")

        if "(" in data_lines[0]:
            self._read_vector_or_tensor_data(data_lines)
        else:
            self._read_scalar_data(data_lines)

        self._validate_data_shape()

    def crop_by_time(self, lower_limit: float = -1e10, upper_limit: float = 1e10):
        """
        Crop planar-average rows to an open time interval in-place.

        Rows are retained when ``lower_limit < time < upper_limit``.

        Args:
            lower_limit: Lower time bound.
            upper_limit: Upper time bound.
        """
        mask = (self.time > lower_limit) & (self.time < upper_limit)

        self.time = self.time[mask]
        self.dt = self.dt[mask]
        self.data = self.data[mask]

    def append(self, other: PlanarAverageFile):
        """
        Append another planar average along the time axis.

        Args:
            other: Planar-average reader to append.

        Raises:
            ValueError: If the two readers use different height coordinates.
        """
        if self.height.shape != other.height.shape or not np.allclose(
            self.height, other.height
        ):
            raise ValueError("Planar average heights do not match.")

        self.time = np.concatenate((self.time, other.time))
        self.dt = np.concatenate((self.dt, other.dt))
        self.data = np.concatenate((self.data, other.data))

    def _parse_height_line(self, line: str) -> np.ndarray:
        values = re.findall(self._FLOAT_PATTERN, line)
        if not values:
            raise ValueError(
                f"The height header in '{self.path}' does not contain any heights."
            )
        return np.array(values, dtype=float)

    def _read_scalar_data(self, data_lines: list[str]) -> None:
        data = np.loadtxt(data_lines)
        data = np.atleast_2d(data)

        if data.shape[1] < 3:
            raise ValueError(
                f"The scalar data in '{self.path}' must contain time, dt, and values."
            )

        self.time = np.array(data[:, 0], dtype=float)
        self.dt = np.array(data[:, 1], dtype=float)
        self.data = np.array(data[:, 2:], dtype=float)

    def _read_vector_or_tensor_data(self, data_lines: list[str]) -> None:
        row_pattern = re.compile(
            rf"^\s*({self._FLOAT_PATTERN})\s+({self._FLOAT_PATTERN})\s+(.*)$"
        )

        time = []
        dt = []
        rows = []
        component_count = None

        for line in data_lines:
            match = row_pattern.match(line)
            if not match:
                raise ValueError(f"Could not parse planar average row: '{line}'.")

            values = re.findall(r"\(([^()]*)\)", match.group(3))
            if not values:
                raise ValueError(f"Could not parse vector/tensor values: '{line}'.")

            row = [
                np.array(re.findall(self._FLOAT_PATTERN, value), dtype=float)
                for value in values
            ]

            if component_count is None:
                component_count = len(row[0])
            if any(len(value) != component_count for value in row):
                raise ValueError(
                    f"Inconsistent vector/tensor component count in '{self.path}'."
                )

            time.append(float(match.group(1)))
            dt.append(float(match.group(2)))
            rows.append(row)

        self.time = np.array(time, dtype=float)
        self.dt = np.array(dt, dtype=float)
        self.data = np.array(rows, dtype=float)

    def _validate_data_shape(self) -> None:
        if self.time.ndim != 1:
            raise ValueError("time must be a one-dimensional array.")
        if self.dt.ndim != 1 or len(self.dt) != len(self.time):
            raise ValueError("dt must be a one-dimensional array matching time.")
        if self.data.shape[0] != len(self.time):
            raise ValueError("data and time must have the same number of rows.")
        if self.data.shape[1] != len(self.height):
            raise ValueError("data must have one value for each height.")
