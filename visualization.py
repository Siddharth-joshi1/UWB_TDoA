# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(points, values, title="Error Heatmap"):
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6,5))
    mask = ~np.isnan(values)

    plt.tricontourf(x[mask], y[mask], values[mask], levels=30)
    plt.colorbar(label="Error (m)")
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.show()

def plot_3d_surface(points, values, title="3D Error Surface"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    x = points[:, 0]
    y = points[:, 1]
    z = values

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Error (m)')

    plt.tight_layout()
    plt.show()