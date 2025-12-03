import numpy as np

def find_amplitude_and_phase_at_frequency(time_array: np.ndarray, data_array: np.ndarray, frequency: float):
    z = np.exp(-2j * np.pi * frequency * time_array)
    C = np.dot(data_array, z)
    
    amplitude = 2.0*np.abs(C) / time_array.size
    phase = np.angle(C)
    
    return amplitude, phase