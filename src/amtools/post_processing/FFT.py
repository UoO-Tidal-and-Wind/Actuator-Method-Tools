# pylint: disable=invalid-name
"""Helpers for extracting frequency-domain properties from sampled data."""

import numpy as np


def find_amplitude_and_phase_at_frequency(
    time_array: np.ndarray, data_array: np.ndarray, frequency: float
):
    """Return the amplitude and phase of ``data_array`` at ``frequency``."""
    z = np.exp(-2j * np.pi * frequency * time_array)
    coefficient = np.dot(data_array, z)

    amplitude = 2.0 * np.abs(coefficient) / time_array.size
    phase = np.angle(coefficient)

    return amplitude, phase
