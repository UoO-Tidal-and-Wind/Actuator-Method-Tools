"""Tests for OpenFOAM planar averaging readers."""

import sys
from types import SimpleNamespace

import numpy as np

# input_output currently imports SurfaceFiles, whose optional dependency is not
# needed by these planar-average tests.
sys.modules.setdefault("pyvista", SimpleNamespace(DataObject=object))

from amtools.input_output.case_reader import CaseReader
from amtools.input_output.post_processing.planar_average_file import PlanarAverageFile


def test_planar_average_file_reads_scalar(tmp_path):
    path = tmp_path / "p"
    path.write_text(
        "\n".join(
            [
                "Height (m) 1.25 2.5 3.75",
                "Time(s) dt (s) p",
                "0 0.1 20 40 30",
                "0.1 0.1 21 41 31",
            ]
        ),
        encoding="utf-8",
    )

    result = PlanarAverageFile(path)
    result.read()

    np.testing.assert_allclose(result.height, [1.25, 2.5, 3.75])
    np.testing.assert_allclose(result.time, [0.0, 0.1])
    np.testing.assert_allclose(result.dt, [0.1, 0.1])
    np.testing.assert_allclose(result.data, [[20, 40, 30], [21, 41, 31]])
    assert result.data.shape == (2, 3)


def test_planar_average_file_reads_vector(tmp_path):
    path = tmp_path / "U"
    path.write_text(
        "\n".join(
            [
                "Height (m) 1 2",
                "Time(s) dt (s) U",
                "0 0.1 (0 1 2) (3 4 5)",
                "0.1 0.1 (6 7 8) (9 10 11)",
            ]
        ),
        encoding="utf-8",
    )

    result = PlanarAverageFile(path)
    result.read()

    np.testing.assert_allclose(result.height, [1.0, 2.0])
    np.testing.assert_allclose(
        result.data,
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ],
    )
    assert result.data.shape == (2, 2, 3)


def test_case_reader_planar_average_combines_time_directories(tmp_path):
    case_path = tmp_path / "case"
    first_path = case_path / "postProcessing" / "planarAveraging" / "0"
    second_path = case_path / "postProcessing" / "planarAveraging" / "1"
    first_path.mkdir(parents=True)
    second_path.mkdir(parents=True)
    (first_path / "p").write_text(
        "\n".join(
            [
                "Height (m) 1 2",
                "Time(s) dt (s) p",
                "0 0.1 10 20",
                "1 0.1 11 21",
            ]
        ),
        encoding="utf-8",
    )
    (second_path / "p").write_text(
        "\n".join(
            [
                "Height (m) 1 2",
                "Time(s) dt (s) p",
                "1 0.1 111 121",
                "2 0.1 12 22",
            ]
        ),
        encoding="utf-8",
    )

    result = CaseReader(case_path).planar_average(
        "planarAveraging", "p", time_dir="combined"
    )

    np.testing.assert_allclose(result.time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.dt, [0.1, 0.1, 0.1])
    np.testing.assert_allclose(result.data, [[10, 20], [11, 21], [12, 22]])
