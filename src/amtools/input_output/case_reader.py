"""
case_reader
===========

This module defines the `CaseReader` class, which provides functionality for reading and
handling case directories related to turbine output and post-processing results.

The `CaseReader` class facilitates the following:
- Accessing turbine output data.
- Retrieving post-processing probe data.
- Navigating through time directories based on specific criteria (latest, first, exactly, closest
    to).

Classes:
    CaseReader: Handles access to turbine output and post-processing data from a case directory.

Example:
    Example usage of the `CaseReader` class:

    ```python
    case = CaseReader("/path/to/case")
    turbine_data = case.turbine_output("turbine_data.dat", time_dir="latest")
    probe_data = case.probe("probe_1", "U", time_dir="first")
    ```
"""

import logging
from pathlib import Path
from typing import Literal
import numpy as np

from .turbine_output.turbine_output_file import TurbineOutputFile
from .post_processing.probe_file import ProbeFile
from .post_processing.surface_files import SurfaceFiles
from .post_processing.force_file import ForceFile
from .post_processing.time_file import TimeFile
from .post_processing.phase_probe_file import PhaseProbeFile

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            raise FileNotFoundError(f"The directory '{root_path}' does not exist.")

        self.name: str = self.path.name  # Direct access to the name of the path
        self.turbine_output_path: Path = self.path / "turbineOutput"
        self.post_processing_path: Path = self.path / "postProcessing"
        self.turbine_output_time_dir = ""

        if not self.turbine_output_path.exists():
            logging.warning(
                "The 'turbineOutput' directory is missing in '%s'.", self.path
            )

        if not self.post_processing_path.exists():
            logging.warning(
                "The 'postProcessing' directory is missing in '%s'.", self.path
            )

    def set_path(self, root_path: str):
        """
        Updates the path for the case directory.

        Args:
            root_path (str): The new path to the root directory of the case.
        """
        self.path = Path(root_path)

    def turbine_output(
        self,
        file_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
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
                - "combine": Attempts to combine time directories together.
                Defaults to "latest".
            time_dir_value (str, optional): Required if `time_dir` is "exactly" or "closest to".
                Specifies the exact or target time value.

        Returns:
            TurbineOutputFile: An instance of the TurbineOutputFile class after reading the file.

        Raises:
            ValueError: If `time_dir` is "exactly" or "closest to" but `time_dir_value`
            is not provided.
        """
        combine_files = False
        if time_dir == "combined":
            # read in the the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

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
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(np.array(time_dirs_float) - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = self.turbine_output_path / self.turbine_output_time_dir / file_name

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = TurbineOutputFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(
                self.turbine_output_time_dir
            )  # remove current time dir

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = self.turbine_output_path / new_time_dir / file_name
                    new_file_reader = TurbineOutputFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_min_time = new_file_reader.time.min()

                    # if new_min_time > current_max_time:
                    #     break  # only combine files that have some overlap in time

                    # crop the new file
                    new_file_reader.crop_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.dt = np.concatenate(
                        (file_reader.dt, new_file_reader.dt)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    file_reader.blade = np.concatenate(
                        (file_reader.blade, new_file_reader.blade)
                    )
                    file_reader.turbine = np.concatenate(
                        (file_reader.turbine, new_file_reader.turbine)
                    )

        return file_reader

    def probe(
        self,
        probe_name: str,
        variable_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Locate and load a probe file for a given variable from a specified time directory.

        Parameters
        ----------
        probe_name : str
            Name of the probe directory within the post-processing output.
        variable_name : str
            Name of the variable to load (e.g., 'U', 'p').
        time_dir : Literal["latest", "first", "exactly", "closest to"], optional
            Strategy for selecting the time directory:
            - "latest": use the highest time value (default)
            - "first": use the lowest time value
            - "exactly": use the directory that exactly matches `time_dir_value`
            - "closest to": use the directory closest to `time_dir_value`
            - "combined": Attempts to combined time directories together.
        time_dir_value : str, optional
            Required if `time_dir` is "exactly" or "closest to". Should be a string representation
            of a floating-point time (e.g., "12.5").

        Returns
        -------
        ProbeFile
            A `ProbeFile` object containing the data read from the selected file.

        Raises
        ------
        FileNotFoundError
            If the specified probe directory or variable file does not exist.
        ValueError
            If `time_dir_value` is not provided when required.
        """

        probe_path = self.post_processing_path / probe_name

        if not probe_path.exists():
            logging.warning("The '%s' path is missing.", probe_path)
            raise FileNotFoundError(f"The path '{probe_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in probe_path.iterdir()]
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
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = probe_path / self.turbine_output_time_dir / variable_name

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = ProbeFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = probe_path / new_time_dir / variable_name
                    new_file_reader = ProbeFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_min_time = new_file_reader.time.min()

                    # if new_min_time > current_max_time:
                    #     break  # only combine files that have some overlap in time

                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    # file_reader.x = np.concatenate((file_reader.x, new_file_reader.x))
                    # file_reader.y = np.concatenate((file_reader.y, new_file_reader.y))
                    # file_reader.z = np.concatenate((file_reader.z, new_file_reader.z))

        return file_reader
    
    def phaseProbe(
        self,
        probe_name: str,
        variable_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Locate and load a probe file for a given variable from a specified time directory.

        Parameters
        ----------
        probe_name : str
            Name of the probe directory within the post-processing output.
        variable_name : str
            Name of the variable to load (e.g., 'U', 'p').
        time_dir : Literal["latest", "first", "exactly", "closest to"], optional
            Strategy for selecting the time directory:
            - "latest": use the highest time value (default)
            - "first": use the lowest time value
            - "exactly": use the directory that exactly matches `time_dir_value`
            - "closest to": use the directory closest to `time_dir_value`
            - "combined": Attempts to combined time directories together.
        time_dir_value : str, optional
            Required if `time_dir` is "exactly" or "closest to". Should be a string representation
            of a floating-point time (e.g., "12.5").

        Returns
        -------
        ProbeFile
            A `ProbeFile` object containing the data read from the selected file.

        Raises
        ------
        FileNotFoundError
            If the specified probe directory or variable file does not exist.
        ValueError
            If `time_dir_value` is not provided when required.
        """

        probe_path = self.post_processing_path / probe_name

        if not probe_path.exists():
            logging.warning("The '%s' path is missing.", probe_path)
            raise FileNotFoundError(f"The path '{probe_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in probe_path.iterdir()]
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
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = probe_path / self.turbine_output_time_dir / variable_name

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = PhaseProbeFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = probe_path / new_time_dir / variable_name
                    new_file_reader = PhaseProbeFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_min_time = new_file_reader.time.min()

                    # if new_min_time > current_max_time:
                    #     break  # only combine files that have some overlap in time

                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    # file_reader.x = np.concatenate((file_reader.x, new_file_reader.x))
                    # file_reader.y = np.concatenate((file_reader.y, new_file_reader.y))
                    # file_reader.z = np.concatenate((file_reader.z, new_file_reader.z))

        return file_reader
    
    def surfaces(self, surfaces_name: str):
        if not (self.post_processing_path / surfaces_name).exists():
            logging.warning("The '%s' path is missing.", self.post_processing_path / surfaces_name)
            raise FileNotFoundError(f"The path '{self.post_processing_path / surfaces_name}' does not exist.")
        surface_files = SurfaceFiles(self.post_processing_path / surfaces_name)
        return surface_files
    
    def force(
        self,
        force_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Locate and load a probe file for a given variable from a specified time directory.

        Parameters
        ----------
        probe_name : str
            Name of the probe directory within the post-processing output.
        variable_name : str
            Name of the variable to load (e.g., 'U', 'p').
        time_dir : Literal["latest", "first", "exactly", "closest to"], optional
            Strategy for selecting the time directory:
            - "latest": use the highest time value (default)
            - "first": use the lowest time value
            - "exactly": use the directory that exactly matches `time_dir_value`
            - "closest to": use the directory closest to `time_dir_value`
            - "combined": Attempts to combined time directories together.
        time_dir_value : str, optional
            Required if `time_dir` is "exactly" or "closest to". Should be a string representation
            of a floating-point time (e.g., "12.5").

        Returns
        -------
        ForceFile
            A `ForceFile` object containing the data read from the selected file.

        Raises
        ------
        FileNotFoundError
            If the specified probe directory or variable file does not exist.
        ValueError
            If `time_dir_value` is not provided when required.
        """

        force_path = self.post_processing_path / force_name

        if not force_path.exists():
            logging.warning("The '%s' path is missing.", force_path)
            raise FileNotFoundError(f"The path '{force_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in force_path.iterdir()]
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
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = force_path / self.turbine_output_time_dir / "force.dat"

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = ForceFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = force_path / new_time_dir / "force.dat"
                    new_file_reader = ForceFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_min_time = new_file_reader.time.min()

                    # if new_min_time > current_max_time:
                    #     break  # only combine files that have some overlap in time

                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    
                    file_reader.split_data()
                    

        return file_reader

    def time(
        self,
        time_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Locate and load a probe file for a given variable from a specified time directory.

        Parameters
        ----------
        probe_name : str
            Name of the probe directory within the post-processing output.
        variable_name : str
            Name of the variable to load (e.g., 'U', 'p').
        time_dir : Literal["latest", "first", "exactly", "closest to"], optional
            Strategy for selecting the time directory:
            - "latest": use the highest time value (default)
            - "first": use the lowest time value
            - "exactly": use the directory that exactly matches `time_dir_value`
            - "closest to": use the directory closest to `time_dir_value`
            - "combined": Attempts to combined time directories together.
        time_dir_value : str, optional
            Required if `time_dir` is "exactly" or "closest to". Should be a string representation
            of a floating-point time (e.g., "12.5").

        Returns
        -------
        TimeFile
            A `TimeFile` object containing the data read from the selected file.

        Raises
        ------
        FileNotFoundError
            If the specified probe directory or variable file does not exist.
        ValueError
            If `time_dir_value` is not provided when required.
        """

        time_path = self.post_processing_path / time_name

        if not time_path.exists():
            logging.warning("The '%s' path is missing.", time_path)
            raise FileNotFoundError(f"The path '{time_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in time_path.iterdir()]
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
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = time_path / self.turbine_output_time_dir / "timeInfo.dat"

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = TimeFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = time_path / new_time_dir / "timeInfo.dat"
                    new_file_reader = TimeFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_min_time = new_file_reader.time.min()

                    # if new_min_time > current_max_time:
                    #     break  # only combine files that have some overlap in time

                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    
                    file_reader.split_data()
                    

        return file_reader