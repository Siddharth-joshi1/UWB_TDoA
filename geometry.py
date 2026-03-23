# geometry.py
import numpy as np

def generate_grid(area_size, resolution):
    x = np.arange(0, area_size, resolution)
    y = np.arange(0, area_size, resolution)
    xv, yv = np.meshgrid(x, y)
    return np.vstack([xv.ravel(), yv.ravel()]).T


def compute_distances(point, anchors):
    return np.linalg.norm(anchors - point, axis=1)