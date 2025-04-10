"""
AnimationControl Module

This module defines the `AnimationControl` class, which allows for the control and configuration of
animation timing.
The class enables users to define how the animation frames are sampled based on time, including
options for:
- Every timestep
- Every n timesteps
- Snapping to specific times
- Using indexed time points

The `AnimationControl` class provides methods for setting output paths, configuring start and end
times, and selecting the time-setting mode.

Imports:
    - List, Optional, Literal (from typing): For type hinting and parameter specification.
    - numpy: For numerical operations, particularly for handling time-related arrays.

Classes:
    - `AnimationControl`: A class for managing animation time settings and frame selection.
"""

from typing import Optional, Literal
import numpy as np


class AnimationControl:
    """
    A class that manages the control and configuration of animation timing.

    Attributes:
        time_setting (str): The mode of time setting. One of the following:
            - "every timestep"
            - "every n timesteps"
            - "snap to times"
            - "indexed"
        snap_times (np.ndarray): An array of specific times to snap to (used with "snap to times").
        step (int): The number of timesteps to skip, used when mode is "every n timesteps".
        time_indicies (np.ndarray): Indices of time points (used with "indexed").
        start_time (float): The start time for the animation.
        end_time (float): The end time for the animation.
        out_path (str): The output path for the animation.

    Methods:
        set_out_path(path):
            Sets the output path for the animation.

        start_and_end_time(start_time, end_time):
            Sets the start and end times for the animation.

        set_time_setting(setting, step, snap_times, indicies):
            Sets the time-setting mode and associated parameters.

        get_frame_time_indicies(t_arr):
            Retrieves the indices of frames based on the selected time-setting mode.
    """

    def __init__(self):
        """
        Initializes the AnimationControl with default parameters.
        """
        self.time_setting = "every timestep"
        self.snap_times = np.array([])
        self.step = -1
        self.time_indicies = np.array([])
        self.start_time = -1e10
        self.end_time = 1e10

        self.out_path = ""

    def set_out_path(self, path):
        """
        Set the output path for the animation.

        Args:
            path (str): The path to store the animation output.
        """
        self.out_path = path

    def start_and_end_time(self, start_time: float = -1e10, end_time: float = 1e10):
        """
        Set the start and end times for the animation.

        Args:
            start_time (float): The start time (default is -1e10).
            end_time (float): The end time (default is 1e10).
        """
        self.start_time = start_time
        self.end_time = end_time

    def set_time_setting(
        self,
        setting: Literal[
            "every timestep", "every n timesteps", "snap to times", "indexed"
        ],
        step: int = 1,
        snap_times: Optional[np.ndarray] = None,
        indicies: Optional[np.ndarray] = None,
    ):
        """
        Set the time-setting mode based on the user's input.

        Args:
            setting (str): One of the following string values:
                - "every timestep"
                - "every n timesteps"
                - "snap to times"
                - "indexed"
            step (int): The number of timesteps to skip when setting is "every n timesteps".
            Default is 1.
            snap_times (Optional[np.ndarray]): A list of times to snap to when setting is "snap
            to times". Default is None.
            indicies (Optional[np.ndarray]): A list of indices to use when setting is "indexed".
            Default is None.

        Raises:
            ValueError: If invalid input is provided for the time-setting mode.
        """
        self.time_setting = setting
        if setting == "every n timesteps" and step < 1:
            raise ValueError("For 'every n timesteps', step must be >= 1.")

        self.step = step
        if setting == "snap to times" and snap_times is None:
            raise ValueError(
                "For 'snap to times', you must provide the snap_times list."
            )

        if setting == "indexed" and indicies is None:
            raise ValueError("For 'indexed', you must provide the indicies list.")

        self.snap_times = snap_times
        self.time_indicies = indicies

    def get_frame_time_indicies(self, t_arr: np.ndarray) -> np.ndarray:
        """
        Get the indices of the frames that correspond to the selected time-setting mode.

        Args:
            t_arr (np.ndarray): The array of times to compare against.

        Returns:
            np.ndarray: A numpy array of indices corresponding to the selected time-setting mode.
        """
        index_arr = np.arange(len(t_arr))

        mask = (t_arr > self.start_time) & (t_arr < self.end_time)
        index_arr = index_arr[mask]

        if self.time_setting == "every timestep":
            index_arr = index_arr[:]
        if self.time_setting == "every n timesteps":
            index_arr = index_arr[:: self.step]
        if self.time_setting == "snap to times":
            inds = np.array(
                [np.argmin(np.abs(t_arr - snap_time)) for snap_time in self.snap_times]
            )
            index_arr = index_arr[inds]
        if self.time_setting == "indexed":
            index_arr = self.time_indicies

        return index_arr
