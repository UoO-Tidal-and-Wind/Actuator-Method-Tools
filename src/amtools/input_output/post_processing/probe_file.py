"""
Readers and writers for OpenFOAM probe output files.

The :class:`ProbeFile` class parses probe coordinates from header lines and
loads scalar, vector, or tensor time-series values into NumPy arrays. Plain text
and gzip-compressed files are supported.
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
    Reader and writer for OpenFOAM probe output.

    Attributes:
        path: Path to the probe output file.
        time: One-dimensional array of output times.
        data: Probe values. Scalars have shape ``(time, probes)``; vector/tensor
            values have shape ``(time, probes, components)``.
        x: Probe x-coordinates.
        y: Probe y-coordinates.
        z: Probe z-coordinates.
        include_out_bounds: Whether to include probes marked ``# Not Found`` while
            parsing the header.
    """

    def __init__(self, file_path: str):
        """
        Initialize an empty probe reader.

        The file is not read until :meth:`read` is called. The path may point to a file
        that does not exist yet when the object will be populated and written with
        :meth:`write`.

        Args:
            file_path: Path to the probe output file.
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
        Update the path to the probe output file.

        Args:
            file_path: New path to use for subsequent reads or writes.
        """
        self.path = file_path

    def read(self):
        """
        Read probe coordinates and time-series values from disk.

        Header lines beginning with ``# Probe`` provide the probe locations. Data rows
        are interpreted as scalar output when they contain plain numeric columns, and as
        vector/tensor output when values are enclosed in parentheses.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            Exception: If the data type cannot be inferred from the file contents.
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
                    if not self.include_out_bounds:
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
            raise ValueError("Could not infer file_type")

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
        Return values for one or more probe indices.

        Args:
            probe_index: Probe index or sequence of probe indices to select.

        Returns:
            The selected slice of ``data``.
        """
        return self.data[:, probe_index]

    def crop_data_by_probe_index(self, probe_index: Union[int, Sequence[int]]):
        """
        Keep only the selected probe indices in-place.

        Args:
            probe_index: Probe index or sequence of probe indices to retain.
        """
        if isinstance(probe_index, int):
            probe_index = [probe_index]

        self.data = self.data[:, probe_index]
        self.x = self.x[probe_index]
        self.y = self.y[probe_index]
        self.z = self.z[probe_index]

    def crop_by_time(self, lower_limit: float = -1e10, upper_limit: float = 1e10):
        """
        Crop probe values to an open time interval in-place.

        Rows are retained when ``lower_limit < time < upper_limit``.

        Args:
            lower_limit: Lower time bound.
            upper_limit: Upper time bound.
        """
        mask = (self.time > lower_limit) & (self.time < upper_limit)

        self.data = self.data[mask]
        self.time = self.time[mask]

    def get_data_by_component(
        self, component_index: Union[int, Sequence[int]]
    ) -> np.ndarray:
        """
        Return selected vector or tensor components.

        Args:
            component_index: Component index or sequence of component indices to select.

        Returns:
            The selected component slice of ``data``.
        """
        return self.data[:, :, component_index]

    def crop_by_component_index(self, component_index: Union[int, Sequence[int]]):
        """
        Keep only selected vector or tensor components in-place.

        Args:
            component_index: Component index or sequence of component indices to retain.
        """
        if isinstance(component_index, int):
            component_index = [component_index]

        self.data = self.data[:, :, component_index]


    def find_nearest_probe(
        self,
        x: Union[float, Sequence[float], np.ndarray],
        y: float | None = None,
        z: float | None = None,
        *,
        dimension: Union[str, int, None] = None,
    ) -> Union[int, np.ndarray]:
        """
        Return the nearest probe index for one or more target coordinates.

        By default, distance is computed in 3D. Set ``dimension`` to ``"x"``,
        ``"y"``, ``"z"``, ``0``, ``1``, or ``2`` to search using only one
        coordinate component.

        Args:
            x: Target x-coordinate when passing separate components, a
                coordinate sequence with shape ``(3,)``, an array of coordinates
                with shape ``(N, 3)``, or a scalar target value when using
                ``dimension``.
            y: Target y-coordinate when passing separate components.
            z: Target z-coordinate when passing separate components.
            dimension: Optional coordinate dimension to use for a one-dimensional
                nearest-probe search.

        Returns:
            Index of the nearest probe for a single target, or an array of
            indices for multiple targets.

        Raises:
            ValueError: If no probe coordinates are loaded, if only one of
                ``y`` or ``z`` is provided, if ``dimension`` is invalid, or if
                the coordinate input shape is not supported.
        """
        dimension_index = self._coerce_dimension(dimension)

        if len(self.x) == 0:
            raise ValueError("No probe coordinates are loaded.")

        probe_coordinates = np.column_stack((self.x, self.y, self.z))
        if dimension_index is None:
            target_coordinates, single_target = self._coerce_target_coordinates(x, y, z)
            distances = np.linalg.norm(
                probe_coordinates[None, :, :] - target_coordinates[:, None, :], axis=2
            )
        else:
            target_values, single_target = self._coerce_target_dimension_values(
                x, y, z, dimension_index
            )
            distances = np.abs(
                probe_coordinates[None, :, dimension_index] - target_values[:, None]
            )

        nearest_indices = np.argmin(distances, axis=1).astype(int)
        if single_target:
            return int(nearest_indices[0])
        return nearest_indices

    @staticmethod
    def _coerce_dimension(dimension: Union[str, int, None]) -> int | None:
        if dimension is None:
            return None

        if isinstance(dimension, bool):
            raise ValueError("dimension must be one of 'x', 'y', 'z', 0, 1, or 2.")

        if isinstance(dimension, str):
            dimension_lookup = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
            try:
                return dimension_lookup[dimension.lower()]
            except KeyError as exc:
                raise ValueError(
                    "dimension must be one of 'x', 'y', 'z', 0, 1, or 2."
                ) from exc

        if dimension in (0, 1, 2):
            return int(dimension)

        raise ValueError("dimension must be one of 'x', 'y', 'z', 0, 1, or 2.")

    @staticmethod
    def _coerce_target_coordinates(
        x: Union[float, Sequence[float], np.ndarray],
        y: float | None,
        z: float | None,
    ) -> tuple[np.ndarray, bool]:
        if y is None and z is None:
            coordinates = np.asarray(x, dtype=float)

            if coordinates.ndim == 1:
                if coordinates.shape[0] != 3:
                    raise ValueError("Coordinate input must have length 3.")
                return coordinates.reshape(1, 3), True

            if coordinates.ndim == 2 and coordinates.shape[1] == 3:
                return coordinates, False

            raise ValueError(
                "Coordinate input must have shape (3,) or (N, 3) for a 3D search."
            )

        if y is None or z is None:
            raise ValueError(
                "y and z must be provided together when passing coordinates separately."
            )

        return np.array([[x, y, z]], dtype=float), True

    @classmethod
    def _coerce_target_dimension_values(
        cls,
        x: Union[float, Sequence[float], np.ndarray],
        y: float | None,
        z: float | None,
        dimension_index: int,
    ) -> tuple[np.ndarray, bool]:
        if y is not None or z is not None:
            coordinates, _ = cls._coerce_target_coordinates(x, y, z)
            return coordinates[:, dimension_index], True

        values = np.asarray(x, dtype=float)

        if values.ndim == 0:
            return values.reshape(1), True

        if values.ndim == 1:
            if values.shape[0] == 3:
                return values[dimension_index : dimension_index + 1], True
            return values, False

        if values.ndim == 2:
            if values.shape[1] == 3:
                return values[:, dimension_index], False
            if values.shape[1] == 1:
                return values[:, 0], False

        raise ValueError(
            "Coordinate input must be a scalar, shape (N,), shape (3,), "
            "shape (N, 1), or shape (N, 3) for a one-dimensional search."
        )

    def append(self, other: ProbeFile):
        """
        Append probe locations from another reader at the same times.

        The operation concatenates probe coordinates and appends ``other.data`` along
        the probe axis. Both readers must contain the same maximum time.

        Args:
            other: Probe reader to append.

        Raises:
            ValueError: If the maximum times differ.
        """
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
        """
        Write the in-memory probe data to disk.

        The output format matches the layout accepted by :meth:`read`. Parent
        directories are created automatically. A ``.gz`` suffix writes compressed text.

        Args:
            file_path: Optional replacement path before writing.

        Raises:
            ValueError: If coordinates, time, and data shapes are inconsistent.
        """
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
