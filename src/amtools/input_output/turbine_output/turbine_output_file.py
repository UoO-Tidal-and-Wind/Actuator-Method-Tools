"""
Reader for files written under an OpenFOAM ``turbineOutput`` directory.
"""

from typing import Union, Sequence
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TurbineOutputFile:
    """
    Reader for turbine output data.

    Attributes:
        path: Path to the turbine output file.
        turbine: Turbine identifier for each row.
        blade: Blade identifier for each row, or ``None`` values when the file has
            no blade column.
        time: One-dimensional array of output times.
        dt: One-dimensional array of timesteps.
        data: Numeric output columns after the metadata columns.
    """

    def __init__(self, file_path: str):
        """
        Initialize a turbine-output reader and validate the path.

        Args:
            file_path: Path to the turbine output file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        self.turbine: np.ndarray = np.array([])
        self.blade: np.ndarray = np.array([])
        self.time: np.ndarray = np.array([])
        self.dt: np.ndarray = np.array([])
        self.data: np.ndarray = np.array([])

    def read(self):
        """
        Read turbine output data from disk.

        The header line supplies metadata columns such as ``Time(s)``, ``dt(s)``,
        ``Turbine``, and optionally ``Blade``. Remaining columns are stored in ``data``.

        Raises:
            ValueError: If the file contains no data rows.
            KeyError: If required metadata columns are missing.
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip().strip("#")
            # Split the header by four spaces
            headers = header_line.split('    ')
        # Read the rest of the file as a DataFrame
        data = pd.read_csv(self.path, delimiter=r'\s+',
                           header=None, comment="#")
        if data.empty:
            logging.warning("No data found in '%s'.", self.path)
            raise ValueError(f"No data found in {self.path}.")

        number_of_headers = len(headers)
        data_key = headers[-1]
        data_columns = data.iloc[:, number_of_headers-1:]
        data_arr = data_columns.to_numpy()
        data = data.iloc[:, :number_of_headers-1]
        headers.remove(data_key)
        data.columns = headers

        try:
            self.time = data["Time(s)"].to_numpy()
            self.dt = data["dt(s)"].to_numpy()
            self.turbine = data["Turbine"].to_numpy()
            self.data = data_arr
        except KeyError as e:
            logging.error("Missing expected column: %s", e)
            raise

        if "Blade" in data.columns:
            self.blade = data["Blade"].to_numpy()
        else:
            self.blade = np.full(self.time.shape, None)

    def set_path(self, file_path: str):
        """
        Update the path to the turbine output file.

        Args:
            file_path: New path to a turbine output file.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def crop_time(self, lower_limit: float = -1E-10, upper_limit: float = 1E10):
        """
        Crop turbine output rows to an open time interval in-place.

        Rows are retained when ``lower_limit < time < upper_limit``.

        Args:
            lower_limit: Lower time bound.
            upper_limit: Upper time bound.
        """
        mask = (self.time > lower_limit) & (self.time < upper_limit)
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]

    def crop_blade(self, blade_index: int):
        """
        Keep only rows for one blade index in-place.

        Args:
            blade_index: Blade identifier to retain.
        """
        mask = self.blade == blade_index
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]

    def crop_turbine(self, turbine_index: int):
        """
        Keep only rows for one turbine index in-place.

        Args:
            turbine_index: Turbine identifier to retain.
        """
        mask = self.turbine == turbine_index
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]

    def get_using_blade_index(self, blade_index: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Return data rows for one or more blade indices.

        Args:
            blade_index: Blade identifier or sequence of blade identifiers to select.

        Returns:
            Rows from ``data`` whose ``blade`` value is selected.
        """
        if isinstance(blade_index, (int, np.integer)):
            blade_index = [blade_index]

        mask = [(blade_ind in blade_index) for blade_ind in self.blade]
        return self.data[mask]
