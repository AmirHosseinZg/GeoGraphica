import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from GeoGraphicaPr.core.computations.functions import convert_seconds


def plot_matrices(
        header_title,
        matrices,
        titles,
        phi_range_deg,
        landa_range_deg,
        start_time,
        saved_path,
        selected_colormap=None,
        contour_levels=None,
        colorbar_min=None,
        colorbar_max=None,
        axis_resolution=0.5,
        grid_size=200
):
    """
    Plots multiple 2D matrices (e.g., total gravity gradients) using contour plots,
    with optional interpolation for smoother visualization.

    ...
    """

    def interpolate_data(matrix, lat_arr, lon_arr):
        """
        Interpolates the given matrix using 'cubic' griddata for smoother contour plots.
        """
        lon_lin = np.linspace(lon_arr.min(), lon_arr.max(), grid_size)
        lat_lin = np.linspace(lat_arr.min(), lat_arr.max(), grid_size)
        lon_new, lat_new = np.meshgrid(lon_lin, lat_lin)

        points = np.array([[lat, lon] for lat in lat_arr for lon in lon_arr])
        values = matrix.flatten()

        smooth_matrix = griddata(points, values, (lat_new, lon_new), method="cubic")
        return lon_new, lat_new, smooth_matrix

    if selected_colormap is None:
        colors = [(0.2, 0.2, 0.8),
                  (0.5, 0.5, 1.0),
                  (1.0, 1.0, 1.0),
                  (1.0, 0.6, 0.2),
                  (0.8, 0.2, 0.2)]
        selected_colormap = LinearSegmentedColormap.from_list("CustomRdOr", colors, N=256)

    if contour_levels is None:
        contour_levels = 10 - 1
    else:
        contour_levels -= 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    for idx, ax in enumerate(axes.flat):
        matrix = matrices[idx]
        title = titles[idx] if idx < len(titles) else f"Matrix {idx + 1}"

        lon_grid, lat_grid, smooth_matrix = interpolate_data(matrix, phi_range_deg, landa_range_deg)

        vmin = colorbar_min if colorbar_min is not None else matrix.min()
        vmax = colorbar_max if colorbar_max is not None else matrix.max()

        levels = np.linspace(vmin, vmax, contour_levels)
        if 0 not in levels:
            levels = np.sort(np.append(levels, 0))

        contour = ax.contourf(lon_grid, lat_grid, smooth_matrix, levels=levels, cmap=selected_colormap)
        contour_lines = ax.contour(lon_grid, lat_grid, smooth_matrix, levels=levels, colors='black', linewidths=0.5)

        ax.set_xticks(np.arange(landa_range_deg.min(), landa_range_deg.max() + axis_resolution, axis_resolution))
        ax.set_yticks(np.arange(phi_range_deg.min(), phi_range_deg.max() + axis_resolution, axis_resolution))
        ax.grid(visible=True, linestyle='-', linewidth=0.5, color='black', alpha=0.7)
        ax.set_aspect('auto')

        ax.set_title(f"{title} (Eö)", fontsize=12)
        ax.set_xlabel("Longitude (deg)", fontsize=10)
        ax.set_ylabel("Latitude (deg)", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        cbar = fig.colorbar(contour, ax=ax, orientation="vertical")
        cbar.set_label("Eötvös Units", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle(header_title, fontsize=16, color="orange", fontweight="bold")

    end_time = time.time()
    total_spent_time = end_time - start_time
    print("Total computation time:", convert_seconds(total_spent_time))

    pdf_path = os.path.join(saved_path, f"{header_title}.pdf")
    png_path = os.path.join(saved_path, f"{header_title}.png")

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)

    plt.close(fig)

    return png_path
