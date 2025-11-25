import pyvista as pv
import numpy as np

class SurfaceFiles:
    def __init__(self, surfaces_data_dir):
        self.data_dir = surfaces_data_dir
        
    
        self.times_paths = [
            p for p in self.data_dir.glob("*")
            if p.name.replace(".", "", 1).isdigit()  # allow numbers like 0.001
        ]
        self.times_names = np.array([path.name for path in self.times_paths])
        self.times_floats = np.array([float(name) for name in self.times_names])
        
        
    def read_file(self, time: float, name: str) -> pv.DataObject:
        if time not in self.times_floats:
            raise ValueError(f"Time {time} not found in available times: {self.times_floats}")
        
        time_index = np.where(self.times_floats == time)[0][0]
        file_path = self.times_paths[time_index]
        
        mesh = pv.read(file_path/name)
        return mesh
    
    def read_nearest_file(self, time: float, name: str) -> pv.DataObject:
        nearest_time = min(self.times_floats, key=lambda t: abs(t - time))
        return self.read_file(nearest_time, name)
    
    def get_time_floats(self) -> np.ndarray:
        return self.times_floats
        
        
        
