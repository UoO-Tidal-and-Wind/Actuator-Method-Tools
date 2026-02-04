from __future__ import annotations
from typing import Union, Sequence
import logging
from pathlib import Path
import re
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

from .probe_file import ProbeFile

class PhaseProbeFile(ProbeFile):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        
        self.bin = np.array([])
        
    def read(self):
        """
        Reads and parses the probe file.
        """
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
        with open(self.path, "r", encoding="utf-8") as file:
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
            (.*)$                     # rest of the line
            """,
            re.VERBOSE,
        )

        data = np.array([])
        # Use regular expressions to split the data lines
        if file_type == "scalar":
            data = np.loadtxt(self.path, comments="#")
        elif file_type == "vector/tensor":
            # Split by 16 spaces and load the data manually
            with open(self.path, "r", encoding="utf-8") as file:
                # Skip header lines
                data_lines = [line for line in file if not line.startswith("#")]

            # Apply regex to split by 16 spaces for each line
            data = np.array([
                [m.group(1), m.group(2), *re.findall(r"\([^()]*\)", m.group(3))]
                for line in data_lines
                if (m := pattern.search(line))
            ], dtype=object)

        self.time = np.array(data[:, 0], dtype=float)
        self.bin = np.array(data[:, 1], dtype=int)
        self.data = data[:, 2:]

        if file_type == "vector/tensor":
            # Initialize the data structure as a list of lists (2D structure)
            self.data = np.array(
                [
                    [np.array(x.strip("()").split(" "), dtype=float) for x in row]
                    for row in self.data
                ]
            )
        
        