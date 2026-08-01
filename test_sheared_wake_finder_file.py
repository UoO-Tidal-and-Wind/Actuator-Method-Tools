"""Tests for sheared-wake-finder readers."""

import sys
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault("pyvista", SimpleNamespace(DataObject=object))

from amtools.input_output.case_reader import CaseReader
from amtools.input_output.wake_finder import ShearedWakeFinderFile


def _write_wake_file(path, rows):
    path.write_text(
        "\n".join(
            [
                "Time(s) centreX(m) centreY(m) centreZ(m) sigmaY(m) sigmaZ(m) Requiv(m)",
                *[" ".join(map(str, row)) for row in rows],
            ]
        ),
        encoding="utf-8",
    )


def test_sheared_wake_finder_file_reads_columns(tmp_path):
    file_path = tmp_path / "wake"
    _write_wake_file(
        file_path,
        [[0.0, 1, 2, 3, 4, 5, 6], [1.0, 7, 8, 9, 10, 11, 12]],
    )

    result = ShearedWakeFinderFile(file_path)
    result.read()

    np.testing.assert_allclose(result.time, [0.0, 1.0])
    np.testing.assert_allclose(result.centre, [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_allclose(result.sigma_y, [4, 10])
    np.testing.assert_allclose(result.sigma_z, [5, 11])
    np.testing.assert_allclose(result.r_equiv, [6, 12])


def test_case_reader_sheared_wake_reader_combines_time_directories(tmp_path):
    case_path = tmp_path / "case"
    first_path = case_path / "shearedWakeFinder" / "0"
    second_path = case_path / "shearedWakeFinder" / "1"
    first_path.mkdir(parents=True)
    second_path.mkdir(parents=True)
    _write_wake_file(
        first_path / "wake", [[0, 1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11, 12]]
    )
    _write_wake_file(
        second_path / "wake",
        [[1, 70, 80, 90, 100, 110, 120], [2, 13, 14, 15, 16, 17, 18]],
    )

    result = CaseReader(case_path).shearedWakeReader("wake", time_dir="combined")

    np.testing.assert_allclose(result.time, [0, 1, 2])
    np.testing.assert_allclose(result.centre, [[1, 2, 3], [7, 8, 9], [13, 14, 15]])
    np.testing.assert_allclose(result.sigma_y, [4, 10, 16])
