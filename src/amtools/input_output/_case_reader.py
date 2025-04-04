"""
_case_reader - Defines the CaseReader class for reading case directories.

This module provides the CaseReader class, which facilitates accessing and
handling data from a specified case directory. It includes functionality
for managing turbine output data and post-processing results.
"""

import logging
from pathlib import Path
from typing import Literal
import numpy as np

from .turbine_output._turbine_output import TurbineOutputFile

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

        file_reader = TurbineOutputFile(file_path)
        file_reader.read()
        return file_reader
