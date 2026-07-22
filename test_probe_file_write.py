"""Round-trip tests for OpenFOAM probe-file writers."""

import sys
from types import SimpleNamespace

import numpy as np

# input_output currently imports SurfaceFiles, whose optional dependency is not
# needed by these probe-file tests.
sys.modules.setdefault("pyvista", SimpleNamespace(DataObject=object))

from amtools.input_output.post_processing.phase_probe_file import PhaseProbeFile
from amtools.input_output.post_processing.probe_file import ProbeFile


def test_probe_file_write_and_read_scalar(tmp_path):
    path = tmp_path / "p"
    probe = ProbeFile(path)
    probe.x = np.array([1.0, 2.0])
    probe.y = np.array([3.0, 4.0])
    probe.z = np.array([5.0, 6.0])
    probe.time = np.array([0.0, 0.5])
    probe.data = np.array([[10.0, 20.0], [30.0, 40.0]])

    probe.write()
    result = ProbeFile(path)
    result.read()

    np.testing.assert_allclose(result.time, probe.time)
    np.testing.assert_allclose(result.data, probe.data)
    np.testing.assert_allclose(result.x, probe.x)
    np.testing.assert_allclose(result.y, probe.y)
    np.testing.assert_allclose(result.z, probe.z)


def test_phase_probe_file_write_and_read_vector_gzip(tmp_path):
    path = tmp_path / "phase-U.gz"
    probe = PhaseProbeFile(path)
    probe.x = np.array([1.0, 2.0])
    probe.y = np.array([3.0, 4.0])
    probe.z = np.array([5.0, 6.0])
    probe.time = np.array([0.0, 0.5])
    probe.bin = np.array([0, 1])
    probe.totalTime = np.array([0.25, 0.75])
    probe.data = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    probe.write()
    result = PhaseProbeFile(path)
    result.read()

    np.testing.assert_allclose(result.time, probe.time)
    np.testing.assert_array_equal(result.bin, probe.bin)
    np.testing.assert_allclose(result.totalTime, probe.totalTime)
    np.testing.assert_allclose(result.data, probe.data)


def test_probe_file_write_and_read_vector(tmp_path):
    path = tmp_path / "U"
    probe = ProbeFile(path)
    probe.x = np.array([1.0])
    probe.y = np.array([2.0])
    probe.z = np.array([3.0])
    probe.time = np.array([0.0, 1.0])
    probe.data = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])

    probe.write()
    result = ProbeFile(path)
    result.read()

    np.testing.assert_allclose(result.data, probe.data)


def test_phase_probe_file_write_and_read_scalar(tmp_path):
    path = tmp_path / "phase-p"
    probe = PhaseProbeFile(path)
    probe.x = np.array([1.0])
    probe.y = np.array([2.0])
    probe.z = np.array([3.0])
    probe.time = np.array([0.0, 1.0])
    probe.bin = np.array([0, 1])
    probe.totalTime = np.array([0.25, 1.25])
    probe.data = np.array([[10.0], [20.0]])

    probe.write()
    result = PhaseProbeFile(path)
    result.read()

    np.testing.assert_allclose(result.time, probe.time)
    np.testing.assert_array_equal(result.bin, probe.bin)
    np.testing.assert_allclose(result.totalTime, probe.totalTime)
    np.testing.assert_allclose(result.data, probe.data)
