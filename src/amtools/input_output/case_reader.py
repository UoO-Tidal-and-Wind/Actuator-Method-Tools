"""
Case-level readers for OpenFOAM actuator-method output.

The :class:`CaseReader` class stores the paths for one simulation case and
provides convenience methods for common output under ``turbineOutput`` and
``postProcessing``. Reader methods support the same time-directory selection
modes where applicable: ``"latest"``, ``"first"``, ``"exactly"``,
``"closest to"``, and ``"combined"``.

Example:
    .. code-block:: python

       case = CaseReader("/path/to/case")
       turbine_data = case.turbine_output("bladeForce", time_dir="latest")
       velocity_probes = case.probe("probes", "U", time_dir="combined")
       planar_u = case.planar_average("planarAveraging", "U", time_dir="combined")
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
from .post_processing.planar_average_file import PlanarAverageFile
from .wake_finder.sheared_wake_finder_file import ShearedWakeFinderFile

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CaseReader:
    """
    Reader facade for one OpenFOAM case directory.

    The class does not load data at construction time. Each public reader method
    locates a specific output file, constructs the corresponding file-reader class,
    loads it, and returns the populated object.

    Attributes:
        path: Root path of the case.
        name: Name of the case directory.
        turbine_output_path: Path to ``turbineOutput`` under the case.
        post_processing_path: Path to ``postProcessing`` under the case.
        turbine_output_time_dir: Name of the most recently selected time directory.
    """

    def __init__(self, root_path: str) -> None:
        """
        Initialize a case reader.

        Args:
            root_path: Path to the root directory of the OpenFOAM case.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
        """
        self.path: Path = Path(root_path)  # Convert to Path object
        if not self.path.exists():
            logging.warning("The '%s' directory is missing.", self.path)
            raise FileNotFoundError(f"The directory '{root_path}' does not exist.")

        self.name: str = self.path.name  # Direct access to the name of the path
        self.turbine_output_path: Path = self.path / "turbineOutput"
        self.post_processing_path: Path = self.path / "postProcessing"
        self.sheared_wake_finder_path: Path = self.path / "shearedWakeFinder"
        self.turbine_output_time_dir = ""
        self.sheared_wake_finder_time_dir = ""

        if not self.turbine_output_path.exists():
            logging.warning(
                "The 'turbineOutput' directory is missing in '%s'.", self.path
            )

        if not self.post_processing_path.exists():
            logging.warning(
                "The 'postProcessing' directory is missing in '%s'.", self.path
            )


        if not self.sheared_wake_finder_path.exists():
            logging.warning(
                "The 'shearedWakeFinder' directory is missing in '%s'.", self.path
            )

    def sheared_wake_reader(
        self,
        object_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ) -> ShearedWakeFinderFile:
        """Load output from a ``shearedWakeFinder`` function object.

        Files are resolved as ``<case>/shearedWakeFinder/<time_dir>/<object_name>``.
        In ``combined`` mode, restart-overlap rows are discarded before appending.
        """
        if not self.sheared_wake_finder_path.exists():
            raise FileNotFoundError(
                f"The path '{self.sheared_wake_finder_path}' does not exist."
            )

        time_dirs = [
            entry.name
            for entry in self.sheared_wake_finder_path.iterdir()
            if entry.is_dir()
        ]
        try:
            time_dirs.sort(key=float)
        except ValueError as exc:
            raise ValueError(
                "All shearedWakeFinder time directories must have numeric names."
            ) from exc
        if not time_dirs:
            raise FileNotFoundError(
                f"No time directories were found in '{self.sheared_wake_finder_path}'."
            )

        combine_files = time_dir == "combined"
        if time_dir in ("first", "combined"):
            selected_time_dir = time_dirs[0]
        elif time_dir == "latest":
            selected_time_dir = time_dirs[-1]
        elif time_dir == "exactly":
            if not time_dir_value:
                raise ValueError("time_dir_value must be provided when using 'exactly'.")
            selected_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError("time_dir_value must be provided when using 'closest to'.")
            selected_time_dir = min(
                time_dirs, key=lambda item: abs(float(item) - float(time_dir_value))
            )
        else:
            raise ValueError(f"Unknown time_dir selection '{time_dir}'.")

        self.sheared_wake_finder_time_dir = selected_time_dir
        file_path = self.sheared_wake_finder_path / selected_time_dir / object_name
        if not file_path.exists():
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = ShearedWakeFinderFile(file_path)
        file_reader.read()

        if combine_files:
            for next_time_dir in time_dirs[1:]:
                next_file_path = (
                    self.sheared_wake_finder_path / next_time_dir / object_name
                )
                if not next_file_path.exists():
                    raise FileNotFoundError(
                        f"The file '{next_file_path}' does not exist."
                    )
                next_reader = ShearedWakeFinderFile(next_file_path)
                next_reader.read()
                next_reader.crop_by_time(lower_limit=file_reader.time.max())
                file_reader.append(next_reader)

        return file_reader

    def set_path(self, root_path: str):
        """
        Update the root path for the case.

        This method only updates ``path``. It does not recalculate cached helper paths
        such as ``turbine_output_path`` or ``post_processing_path``.

        Args:
            root_path: New path to the case root directory.
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
        Load a file from ``turbineOutput``.

        Files are resolved as ``<case>/turbineOutput/<time_dir>/<file_name>``. When
        ``time_dir="combined"``, data are read from the earliest time directory first
        and then concatenated with later directories after dropping rows that overlap
        with the current maximum time.

        Args:
            file_name: Name of the turbine output file to load.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated
            :class:`~amtools.input_output.turbine_output.turbine_output_file.TurbineOutputFile`.

        Raises:
            FileNotFoundError: If the resolved output file does not exist.
            ValueError: If ``time_dir_value`` is required but not provided.
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
        include_out_of_bounds=False,
    ):
        """
        Load an OpenFOAM probe variable from ``postProcessing``.

        Files are resolved as ``<case>/postProcessing/<probe_name>/<time_dir>/<variable_name>``.
        If the plain file is missing, a ``.gz`` file with the same name is used when
        available. Scalar data are stored with shape ``(time, probes)`` and
        vector/tensor data with shape ``(time, probes, components)``.

        Args:
            probe_name: Name of the probe directory under ``postProcessing``.
            variable_name: Name of the variable file to load, such as ``"U"`` or ``"p"``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.
            include_out_of_bounds: Whether to keep probes marked ``# Not Found`` in the
                file header.

        Returns:
            A populated :class:`~amtools.input_output.post_processing.probe_file.ProbeFile`.

        Raises:
            FileNotFoundError: If the probe directory or resolved variable file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
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
        if not file_path.exists() and file_path.with_name(f"{file_path.name}.gz").exists():
            file_path = file_path.with_name(f"{file_path.name}.gz")

        if not file_path.exists():
            logging.warning("The '%s' file is missing.", file_path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = ProbeFile(file_path)
        file_reader.include_out_bounds = include_out_of_bounds
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
                    if not new_file_path.exists() and new_file_path.with_name(
                        f"{new_file_path.name}.gz"
                    ).exists():
                        new_file_path = new_file_path.with_name(f"{new_file_path.name}.gz")
                    new_file_reader = ProbeFile(new_file_path)
                    new_file_reader.include_out_bounds = include_out_of_bounds
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
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

    def phase_probe(
        self,
        probe_name: str,
        variable_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Load an OpenFOAM phase-probe variable from ``postProcessing``.

        Files are resolved as ``<case>/postProcessing/<probe_name>/<time_dir>/<variable_name>``.
        The returned reader includes the probe coordinates plus phase-bin metadata
        stored in ``bin`` and ``totalTime``.

        Args:
            probe_name: Name of the phase-probe directory under ``postProcessing``.
            variable_name: Name of the variable file to load.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated
            :class:`~amtools.input_output.post_processing.phase_probe_file.PhaseProbeFile`.

        Raises:
            FileNotFoundError: If the phase-probe directory or resolved variable file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
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
        if not file_path.exists() and file_path.with_name(f"{file_path.name}.gz").exists():
            file_path = file_path.with_name(f"{file_path.name}.gz")

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
                    if not new_file_path.exists() and new_file_path.with_name(
                        f"{new_file_path.name}.gz"
                    ).exists():
                        new_file_path = new_file_path.with_name(f"{new_file_path.name}.gz")
                    new_file_reader = PhaseProbeFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )
                    file_reader.bin = np.concatenate(
                            (file_reader.bin, new_file_reader.bin)
                    )
                    # file_reader.x = np.concatenate((file_reader.x, new_file_reader.x))
                    # file_reader.y = np.concatenate((file_reader.y, new_file_reader.y))
                    # file_reader.z = np.concatenate((file_reader.z, new_file_reader.z))

        return file_reader

    def planar_average(
        self,
        planar_average_name: str,
        variable_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Load a planar averaging variable from ``postProcessing``.

        Files are resolved as
        ``<case>/postProcessing/<planar_average_name>/<time_dir>/<variable_name>``. If
        the plain file is missing, a ``.gz`` file with the same name is used when
        available. Scalar output is stored in ``data`` with shape ``(time, heights)``;
        vector or tensor output is stored with shape ``(time, heights, components)``.

        Args:
            planar_average_name: Name of the planar averaging directory under
                ``postProcessing``.
            variable_name: Name of the variable file to load, such as ``"U"`` or ``"p"``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated
            :class:`~amtools.input_output.post_processing.planar_average_file.PlanarAverageFile`.

        Raises:
            FileNotFoundError: If the planar averaging directory or variable file is missing.
            ValueError: If ``time_dir_value`` is required but not provided, or if an
                unknown ``time_dir`` selector is supplied.
        """

        planar_average_path = self.post_processing_path / planar_average_name

        if not planar_average_path.exists():
            logging.warning("The %s path is missing.", planar_average_path)
            raise FileNotFoundError(
                f"The path {planar_average_path} does not exist."
            )

        combine_files = False
        if time_dir == "combined":
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in planar_average_path.iterdir()]
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
                    "time_dir_value must be provided when using exactly."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using closest to."
                )
            time_dirs_float = np.array([float(f) for f in time_dirs])
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]
        else:
            raise ValueError(f"Unknown time_dir selection {time_dir}.")

        file_path = planar_average_path / self.turbine_output_time_dir / variable_name
        if not file_path.exists() and file_path.with_name(f"{file_path.name}.gz").exists():
            file_path = file_path.with_name(f"{file_path.name}.gz")

        if not file_path.exists():
            logging.warning("The %s file is missing.", file_path)
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        file_reader = PlanarAverageFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = planar_average_path / new_time_dir / variable_name
                    if not new_file_path.exists() and new_file_path.with_name(
                        f"{new_file_path.name}.gz"
                    ).exists():
                        new_file_path = new_file_path.with_name(
                            f"{new_file_path.name}.gz"
                        )

                    new_file_reader = PlanarAverageFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    new_file_reader.crop_by_time(lower_limit=current_max_time)
                    file_reader.append(new_file_reader)

        return file_reader

    def planar_average_legacy(
        self,
        planar_average_name: str,
        variable_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Compatibility wrapper for :meth:`planar_average`.

        Args:
            planar_average_name: Name of the planar averaging directory under
                ``postProcessing``.
            variable_name: Name of the variable file to load.
            time_dir: Time-directory selection mode.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated
            :class:`~amtools.input_output.post_processing.planar_average_file.PlanarAverageFile`.
        """
        return self.planar_average(
            planar_average_name, variable_name, time_dir, time_dir_value
        )

    def surfaces(self, surfaces_name: str):
        """
        Return a surface-file helper for a ``postProcessing`` surface directory.

        Args:
            surfaces_name: Name of the surface output directory under ``postProcessing``.

        Returns:
            A :class:`~amtools.input_output.post_processing.surface_files.SurfaceFiles`
            object for loading files from available surface time directories.

        Raises:
            FileNotFoundError: If the surface output directory is missing.
        """
        if not (self.post_processing_path / surfaces_name).exists():
            logging.warning(
                "The '%s' path is missing.", self.post_processing_path / surfaces_name
            )
            raise FileNotFoundError(
                f"The path '{self.post_processing_path / surfaces_name}' does not exist."
            )
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
        Load ``force.dat`` from a post-processing force directory.

        Files are resolved as ``<case>/postProcessing/<force_name>/<time_dir>/force.dat``.
        When ``time_dir="combined"``, later time directories are appended after rows
        that overlap the current maximum time are dropped.

        Args:
            force_name: Name of the force output directory under ``postProcessing``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated :class:`~amtools.input_output.post_processing.force_file.ForceFile`.

        Raises:
            FileNotFoundError: If the force directory or ``force.dat`` file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
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
        Load ``timeInfo.dat`` from a post-processing time directory.

        Files are resolved as ``<case>/postProcessing/<time_name>/<time_dir>/timeInfo.dat``.
        When ``time_dir="combined"``, later time directories are appended after rows
        that overlap the current maximum time are dropped.

        Args:
            time_name: Name of the time output directory under ``postProcessing``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated :class:`~amtools.input_output.post_processing.time_file.TimeFile`.

        Raises:
            FileNotFoundError: If the time output directory or ``timeInfo.dat`` file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
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
