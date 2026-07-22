"""Tests for nearest-probe lookup helpers."""

import sys
from types import SimpleNamespace

import numpy as np

# input_output currently imports SurfaceFiles, whose optional dependency is not
# needed by these nearest-probe tests.
sys.modules.setdefault("pyvista", SimpleNamespace(DataObject=object))

from amtools.input_output.post_processing.phase_probe_file import PhaseProbeFile
from amtools.input_output.post_processing.probe_file import ProbeFile


def _probe_file_with_coordinates():
    probe = ProbeFile("unused")
    probe.x = np.array([0.0, 10.0, 0.0])
    probe.y = np.array([0.0, 0.0, 5.0])
    probe.z = np.array([0.0, 0.0, 10.0])
    return probe


def _phase_probe_file_with_coordinates():
    probe = PhaseProbeFile("unused")
    probe.x = np.array([0.0, 10.0, 0.0])
    probe.y = np.array([0.0, 0.0, 5.0])
    probe.z = np.array([0.0, 0.0, 10.0])
    return probe


def test_find_nearest_probe_accepts_coordinate_input_forms():
    probe = _probe_file_with_coordinates()

    assert probe.find_nearest_probe(8.0, 0.0, 0.0) == 1
    assert probe.find_nearest_probe((0.0, 4.5, 9.0)) == 2
    assert probe.find_nearest_probe(np.array([0.2, 0.1, 0.0])) == 0

    indices = probe.find_nearest_probe(np.array([[9.0, 0.0, 0.0], [0.0, 4.0, 8.0]]))
    np.testing.assert_array_equal(indices, [1, 2])


def test_find_nearest_probe_can_search_single_dimension():
    probe = _probe_file_with_coordinates()

    assert probe.find_nearest_probe(4.9, dimension="y") == 2
    assert probe.find_nearest_probe((0.0, 0.0, 9.1), dimension="z") == 2
    assert probe.find_nearest_probe(9.0, dimension=0) == 1

    indices = probe.find_nearest_probe(
        np.array([[9.0, 0.0, 0.0], [0.0, 4.9, 0.0]]), dimension=1
    )
    np.testing.assert_array_equal(indices, [0, 2])


def test_phase_probe_find_nearest_uses_probe_coordinate_helpers():
    probe = _phase_probe_file_with_coordinates()

    assert probe.find_nearest_probe((0.0, 4.5, 9.0)) == 2
    np.testing.assert_array_equal(
        probe.find_nearest_probe(np.array([[9.0, 0.0, 0.0], [0.0, 4.0, 8.0]])),
        [1, 2],
    )
    assert probe.find_nearest_probe(4.9, dimension="y") == 2
