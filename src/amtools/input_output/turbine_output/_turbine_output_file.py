"""
_turbine_output_file - defines the TurbineOutputFile class.

This module provides a class for reading turbine output files, extracting 
structured numerical data, and storing it in NumPy arrays.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TurbineOutputFile:
    """
    A class for reading and processing turbine output files.

    Attributes:
        path (str): Path to the turbine output file.
        turbine (np.ndarray): Array containing turbine IDs.
        blade (np.ndarray): Array containing blade IDs (if available).
        time (np.ndarray): Array containing time values.
        dt (np.ndarray): Array containing time step values.
        data (np.ndarray): Array containing numerical data.
    """

    def __init__(self, file_path: str):
        """
        Initializes the TurbineOutputFile instance and verifies the file exists.

        Args:
            file_path (str): Path to the turbine output file.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
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
        Reads and processes the turbine output file.

        This method reads the header and data from the file, processes them, 
        and stores the results in instance attributes.

        Raises:
            ValueError: If the file is empty or contains no valid data.
            KeyError: If expected columns (e.g., "Time(s)", "dt(s)", "Turbine") are missing.
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
        Changes the path to the file.

        Args:
            file_path (str): Path to turbineOutput file.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def crop_time(self, lower_limit: float=-1E-10, upper_limit: float=1E10):
        """
        Filters data based on the time range between `lower_limit` and `upper_limit`.

        Parameters:
        ----------
        lower_limit : float, optional
            The minimum time value to include (default is -1E-10).
        
        upper_limit : float, optional
            The maximum time value to include (default is 1E10).

        Modifies:
        --------
        self.data, self.dt, self.time, self.blade, self.turbine
            These attributes are filtered to include only values within the specified time range.
        """

        mask = (self.time > lower_limit) & (self.time < upper_limit)
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]

    def crop_blade(self, blade_index: int):
        """
        Filters data to include only the specified blade index.

        Parameters:
        ----------
        blade_index : ind
            The index of the blade to keep in the filtered data.

        Modifies:
        --------
        self.data, self.dt, self.time, self.blade, self.turbine
            These attributes are filtered to include only values corresponding to the 
            specified blade.
        """
        mask = self.blade == blade_index
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]

    def crop_turbine(self, turbine_index: int):
        """
        Filters data to include only the specified turbine index.

        Parameters:
        ----------
        turbine_index : ind
            The index of the turbine to keep in the filtered data.

        Modifies:
        --------
        self.data, self.dt, self.time, self.blade, self.turbine
            These attributes are filtered to include only values corresponding to the 
            specified turbine.
        """
        mask = self.turbine == turbine_index
        self.data = self.data[mask]
        self.dt = self.dt[mask]
        self.time = self.time[mask]
        self.blade = self.blade[mask]
        self.turbine = self.turbine[mask]
