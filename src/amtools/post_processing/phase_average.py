"""
Phase Averaging Module
=======================

This module contains the `PhaseAverageResult` class for storing phase-averaged results, 
and the `phase_average_array` function to compute phase averages from time and value arrays.

The `phase_average_array` function bins data based on calculated phase angles and computes 
the mean and standard deviation for each bin. There are options for phase offset removal 
and custom binning.

Classes:
    PhaseAverageResult: Stores phase-averaged data, including mean, standard deviation, 
    and bin counts.

Functions:
    phase_average_array: Computes phase averages for time-series data and returns a 
    `PhaseAverageResult` object containing the results.

Example:
    Example usage of the `phase_average_array` function:

    ```python
    t_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    y_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = phase_average_array(t_arr, y_arr, frequency=1.0)
    print(result.phase_averaged_mean)
    ```
    
"""


import numpy as np


class PhaseAverageResult:
    """
    A class to hold the results of a phase averaging operation.

    Attributes:
        bin_midpoints (np.ndarray): Midpoints of the bins used for phase averaging.
        phase_averaged_mean (np.ndarray): The mean of the phase averaged values for each bin.
        phase_averaged_std (np.ndarray): The standard deviation of the phase averaged values 
        for each bin.
        bin_counts (np.ndarray): The number of values that were assigned to each bin.
    """

    def __init__(self):
        """
        Initializes a PhaseAverageResult object with empty arrays for the attributes.
        """
        self.bin_midpoints = np.array([])
        self.phase_averaged_mean = np.array([])
        self.phase_averaged_std = np.array([])
        self.bin_counts = np.array([])

    def get_bins(self):
        """
        Returns the midpoints of the bins used for phase averaging.

        Returns:
            np.ndarray: Array of bin midpoints.
        """
        return self.bin_midpoints/360
    
    def get_bins_in_degrees(self):
        """
        Returns the midpoints of the bins used for the phase averaging in degrees.
        
        Returns:
            np.ndarray: Array of bin midpoints (degrees)
        """
        return self.bin_midpoints

    def get_mean(self):
        """
        Returns the phase averaged mean for each bin.

        Returns:
            np.ndarray: Array of phase averaged means.
        """
        return self.phase_averaged_mean

    def get_std(self):
        """
        Returns the phase averaged standard deviation for each bin.

        Returns:
            np.ndarray: Array of phase averaged standard deviations.
        """
        return self.phase_averaged_std

    def get_variance(self):
        """
        Returns the phase averaged variance for each bin (square of standard deviation).

        Returns:
            np.ndarray: Array of phase averaged variances.
        """
        return np.square(self.phase_averaged_std)


def phase_average_array(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    frequency: float,
    phase_offset: float = 0,
    number_of_bins: int = 45,
    bin_center_offset: float = None,
    include_0_and_360: bool = True,
    remove_phase_offset: bool = False,
) -> PhaseAverageResult:
    """
    Perform phase averaging on an array of values (`y_arr`) using the corresponding time
    values (`t_arr`).
    This function computes the phase averages for a specified frequency and returns a result object.

    Args:
        t_arr (np.ndarray): Array of time values corresponding to the data in `y_arr`.
        y_arr (np.ndarray): Array (or arrays) of values to phase average. Each row corresponds to a
            separate array of values.
        frequency (float): The frequency at which to phase average the data (in Hz).
        phase_offset (float, optional): The phase offset to apply when t=0 (in degrees). Defaults
            to 0.
        number_of_bins (int, optional): The number of bins to use for phase averaging. Defaults
            to 45.
        bin_center_offset (float, optional): A positive value to shift the bin midpoints. Defaults
            to half the bin width.
        include_0_and_360 (bool, optional): If `True`, include both 0 and 360 as bin midpoints.
            Defaults to `True`.
        remove_phase_offset (bool, optional): If `True`, removes the phase offset by correcting for
            the phase of the signal before averaging. Defaults to `False`.

    Returns:
        PhaseAverageResult: An object containing the phase-averaged results, including the bin
            midpoints, mean values, standard deviation, and bin counts.

    Raises:
        ValueError: If `bin_center_offset` is negative or greater than the allowable bin width.
    """

    result = PhaseAverageResult()

    # Calculate phase array
    phase_arr = (np.degrees(2 * np.pi * frequency * t_arr) + phase_offset) % 360
    bins = np.linspace(0, 360, number_of_bins + 1)
    bin_midpoints = (bins[1:] + bins[0:-1]) * 0.5

    if bin_center_offset is None:
        bin_center_offset = 180 / number_of_bins

    if bin_center_offset != 0:
        if bin_center_offset < 0:
            raise ValueError("Please use a positive bin_center_offset")
        if bin_center_offset > 360.0 / (number_of_bins - 1):
            raise ValueError("bin_center_offset is larger than the bin width")

        # Adjust bin midpoints and bins based on offset
        bins = bins + bin_center_offset
        bin_midpoints = (bin_midpoints + bin_center_offset) % 360

        # Ensure first and last values are 0 and 360
        bins = np.insert(bins, 0, 0)
        bins[-1] = 360

    if remove_phase_offset:
        binned_y_arr = []

        # Calculate phase angle using FFT
        # fs = 1.0 / np.mean(np.diff(t_arr))
        frequencies = np.fft.fftfreq(len(y_arr), np.mean(np.diff(t_arr)))
        fft_values = np.fft.fft(y_arr)
        index = np.argmax(np.abs(frequencies - frequency) < 1e-3)
        y_arr_phase = np.degrees(np.angle(fft_values[index])) + 90

        # Bin values based on corrected phase
        bin_inds = np.digitize(np.mod(phase_arr + y_arr_phase, 360), bins)

        # Create binned data
        binned_y_arr = [y_arr[bin_inds == i] for i in range(1, number_of_bins + 2)]

    else:
        # Bin values based on original phase
        bin_inds = np.digitize(phase_arr, bins)

        # Create binned data
        binned_y_arr = [y_arr[bin_inds == i] for i in range(1, number_of_bins + 2)]

    # Adjust the first and last bins if there is an offset
    if bin_center_offset != 0:
        binned_y_arr[-1] = np.concatenate((binned_y_arr[0], binned_y_arr[-1]))
        binned_y_arr = binned_y_arr[1:]

    # Calculate mean and std for each bin
    phase_averaged_y_arr_std = np.array(
        [np.std(group) if len(group) > 0 else 0 for group in binned_y_arr]
    )
    phase_averaged_y_arr = np.array(
        [np.mean(group) if len(group) > 0 else 0 for group in binned_y_arr]
    )

    # Sort bin midpoints and results
    sorted_bin_midpoints_inds = np.argsort(bin_midpoints)
    bin_midpoints = bin_midpoints[sorted_bin_midpoints_inds]
    phase_averaged_y_arr_std = phase_averaged_y_arr_std[sorted_bin_midpoints_inds]
    phase_averaged_y_arr = phase_averaged_y_arr[sorted_bin_midpoints_inds]

    # Optionally include 0 and 360 as bin midpoints
    if include_0_and_360:
        if 0 in bin_midpoints:
            bin_midpoints = np.append(bin_midpoints, 360)
            phase_averaged_y_arr = np.append(
                phase_averaged_y_arr, phase_averaged_y_arr[0]
            )
            phase_averaged_y_arr_std = np.append(
                phase_averaged_y_arr_std, phase_averaged_y_arr_std[0]
            )
            binned_y_arr.append(binned_y_arr[0])

    # Store the results in the result object
    result.bin_midpoints = np.array(bin_midpoints)
    result.phase_averaged_mean = np.array(phase_averaged_y_arr)
    result.phase_averaged_std = np.array(phase_averaged_y_arr_std)
    result.bin_counts = np.array([len(arr) for arr in binned_y_arr])

    return result
