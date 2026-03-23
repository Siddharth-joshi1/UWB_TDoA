# crlb_map.py
import numpy as np
from config import AREA_SIZE, GRID_RES, ANCHORS, NOISE_STD
from geometry import generate_grid
from crlb import compute_crlb
from visualization import plot_heatmap, plot_3d_surface

points = generate_grid(AREA_SIZE, GRID_RES)

crlb_values = []

for p in points:
    val = compute_crlb(p, ANCHORS, NOISE_STD)
    crlb_values.append(val)

crlb_values = np.array(crlb_values)

plot_heatmap(points, crlb_values, title="CRLB Heatmap")
plot_3d_surface(points, crlb_values, title="CRLB Surface")