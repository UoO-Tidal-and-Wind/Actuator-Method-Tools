"""
Reader and writer support for OpenFOAM phase-probe output files.
"""

from __future__ import annotations
import gzip
import logging
from pathlib import Path
import re
import numpy as np
from .probe_file import ProbeFile

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PhaseProbeFile(ProbeFile):
    """
    Probe reader for phase-averaged probe output.

    This class extends :class:`~amtools.input_output.post_processing.probe_file.ProbeFile`
    with the additional columns written by phase probes.

    Attributes:
        bin: One-dimensional integer array of phase-bin indices.
        total_time: One-dimensional array containing accumulated time per phase-bin sample.
    """
    # Adds phase metadata to ProbeFile's time, data, and coordinate arrays.
    # pylint: disable=too-many-instance-attributes
    def __init__(self, file_path: str):
        """
        Initialize an empty phase-probe reader.

        Args:
            file_path: Path to the phase-probe output file.
        """
        super().__init__(file_path)

        self.bin = np.array([])
        self.total_time = np.array([])
        self.totalTime = self.total_time  # pylint: disable=invalid-name

    # Existing x/y/z, time, and data attributes are populated by ProbeFile.read.
    # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

    def read(self):
        """
        Read phase-probe coordinates, phase metadata, and values from disk.

        Header lines beginning with ``# PhaseProbe`` provide probe locations. Data rows
        must contain ``time``, ``bin``, ``totalTime``, and then one scalar or
        vector/tensor value per probe.

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
            rf"# PhaseProbe (\d+) \(({float_pattern}) ({float_pattern}) ({float_pattern})\)"
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
    (\d+)                     # bin (int)
    \s+
    ([+-]?\d+(?:\.\d+)?)      # totalTime
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
            data = np.array([
                [m.group(1), m.group(2), m.group(3), *re.findall(r"\([^()]*\)", m.group(4))]
                for line in data_lines
                if (m := pattern.search(line))
            ], dtype=object)
        else:
            raise ValueError("Could not infer file_type")

        self.time = np.array(data[:, 0], dtype=float)
        self.bin = np.array(data[:, 1], dtype=int)
        self.total_time = np.array(data[:, 2], dtype=float)
        self.totalTime = self.total_time  # pylint: disable=invalid-name
        self.data = data[:, 3:]

        if file_type == "vector/tensor":
            # Initialize the data structure as a list of lists (2D structure)
            self.data = np.array(
                [
                    [np.array(x.strip("()").split(" "), dtype=float) for x in row]
                    for row in self.data
                ]
            )




    def _probe_header_label(self) -> str:
        return "PhaseProbe"

    def _validate_for_write(self) -> None:
        super()._validate_for_write()
        self.bin = np.asarray(self.bin)
        self.total_time = np.asarray(self.total_time)
        self.totalTime = self.total_time  # pylint: disable=invalid-name
        if self.bin.ndim != 1 or len(self.bin) != len(self.time):
            raise ValueError("bin must be a one-dimensional array matching time.")
        if self.total_time.ndim != 1 or len(self.total_time) != len(self.time):
            raise ValueError("totalTime must be a one-dimensional array matching time.")
        if not np.issubdtype(self.bin.dtype, np.integer):
            if not np.all(np.equal(self.bin, np.floor(self.bin))):
                raise ValueError("bin values must be integers.")

    def _row_prefix(self, row_index: int) -> list[str]:
        return [
            self._format_number(self.time[row_index]),
            str(int(self.bin[row_index])),
            self._format_number(self.total_time[row_index]),
        ]
