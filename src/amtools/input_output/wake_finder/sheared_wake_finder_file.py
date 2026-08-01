"""Reader for tabular output produced by the ``shearedWakeFinder`` function object."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ShearedWakeFinderFile:
    """Read one ``shearedWakeFinder`` output file.

    Input files have one header row followed by the seven whitespace-separated
    columns ``Time(s)``, centre coordinates, ``sigmaY(m)``, ``sigmaZ(m)``, and
    ``Requiv(m)``.
    """

    _EXPECTED_COLUMNS = 7

    def __init__(self, file_path: str | Path) -> None:
        self.path = Path(file_path)
        self.time = np.array([], dtype=float)
        self.centre = np.empty((0, 3), dtype=float)
        self.sigma_y = np.array([], dtype=float)
        self.sigma_z = np.array([], dtype=float)
        self.r_equiv = np.array([], dtype=float)

    def set_path(self, file_path: str | Path) -> None:
        """Update the path used for subsequent reads."""
        self.path = Path(file_path)

    def read(self) -> None:
        """Read numeric data from :attr:`path`."""
        if not self.path.exists():
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")
        try:
            data = np.loadtxt(self.path, skiprows=1, dtype=float, ndmin=2)
        except ValueError as exc:
            raise ValueError(
                f"The sheared-wake file '{self.path}' must contain numeric data."
            ) from exc
        if data.shape[1] != self._EXPECTED_COLUMNS:
            raise ValueError(
                f"The sheared-wake file '{self.path}' has {data.shape[1]} columns; "
                f"expected {self._EXPECTED_COLUMNS}."
            )

        self.time = data[:, 0]
        self.centre = data[:, 1:4]
        self.sigma_y = data[:, 4]
        self.sigma_z = data[:, 5]
        self.r_equiv = data[:, 6]

    def crop_by_time(
        self, lower_limit: float = -np.inf, upper_limit: float = np.inf
    ) -> None:
        """Keep rows in the open interval ``lower_limit < time < upper_limit``."""
        mask = (self.time > lower_limit) & (self.time < upper_limit)
        self.time = self.time[mask]
        self.centre = self.centre[mask]
        self.sigma_y = self.sigma_y[mask]
        self.sigma_z = self.sigma_z[mask]
        self.r_equiv = self.r_equiv[mask]


    def get_data_at_nearest_time(
        self, target_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return wake data for the recorded time nearest to ``target_time``.

        A sheared-wake file can contain multiple rows at one output time. All of
        those rows are returned, in their original file order.

        Returns:
            A tuple of ``(centre, sigma_y, sigma_z, r_equiv)`` arrays for all
            rows at the nearest recorded time.

        Raises:
            ValueError: If no data have been read.
        """
        if self.time.size == 0:
            raise ValueError("No sheared-wake data are loaded.")

        nearest_time = self.time[np.argmin(np.abs(self.time - target_time))]
        mask = self.time == nearest_time
        return (
            self.centre[mask],
            self.sigma_y[mask],
            self.sigma_z[mask],
            self.r_equiv[mask],
        )

    def append(self, other: "ShearedWakeFinderFile") -> None:
        """Append all rows from another sheared-wake reader."""
        self.time = np.concatenate((self.time, other.time))
        self.centre = np.concatenate((self.centre, other.centre), axis=0)
        self.sigma_y = np.concatenate((self.sigma_y, other.sigma_y))
        self.sigma_z = np.concatenate((self.sigma_z, other.sigma_z))
        self.r_equiv = np.concatenate((self.r_equiv, other.r_equiv))
