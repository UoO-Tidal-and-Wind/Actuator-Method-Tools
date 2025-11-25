"""
_force_file
Defines the 'ForceFile' class
"""
    
import logging
from pathlib import Path
import numpy as np
import re
    
class ForceFile:
    def __init__(self, file_path: str):
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        self.time = np.array([])
        self.data = np.array([])
        
        self.total = np.array([])
        self.pressure = np.array([])
        self.viscous = np.array([])
        
    def set_path(self, file_path: str):
        """
        Updates the file path to a new probe output file.

        Args:
            file_path (str): Path to the new probe file.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
        """
        self.path = file_path
        if not Path(self.path).exists():
            logging.error("File '%s' not found.", self.path)
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
    def read(self):
        
        # assumes order of data is total, pressure, viscous
        data = np.genfromtxt(self.path, comments="#")
        self.time = data[:,0]
        self.data = data[:,1:]
        self.split_data()
        
    def split_data(self):
        self.total = self.data[:,0:3]
        self.ptressure = self.data[:,3:6]
        self.viscous = self.data[:,6:9]
        
    def crop_by_time(self, lower_limit: float = None, upper_limit: float = None):
        """
        Crops the data based on time limits.

        Args:
            lower_limit (float, optional): Lower time limit. Defaults to None.
            upper_limit (float, optional): Upper time limit. Defaults to None.
        """
        time_mask = np.ones_like(self.time, dtype=bool)
        
        if lower_limit is not None:
            time_mask &= self.time >= lower_limit
        if upper_limit is not None:
            time_mask &= self.time <= upper_limit
            
        self.time = self.time[time_mask]
        self.data = self.data[time_mask, :]
        self.split_data()
        
        
            
        
        
