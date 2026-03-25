# geometry.py
import numpy as np

def generate_grid(area_size,height, resolution):
    x = np.arange(0, area_size, resolution)
    y = np.arange(0, area_size, resolution)
    z = np.arange(0, height, resolution)

    xv, yv,zv = np.meshgrid(x, y,z)
    return np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T


def compute_distances(point, anchors):
    return np.linalg.norm(anchors - point, axis=1)