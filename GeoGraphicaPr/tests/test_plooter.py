import os
import time
import numpy as np

from GeoGraphicaPr.core.plotting import plotter

import pandas as pd

from GeoGraphicaPr.utils import files_paths
from GeoGraphicaPr.utils.files_paths import base_path


def load_data(file_name):
    data = pd.read_csv(file_name, header=None).to_numpy()
    matrix = data[1:, 1:]  # The actual matrix
    return np.array(matrix)


def test():
    TxxParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxxParker.csv")
    TxyParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxyParker.csv")
    TxzParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxzParker.csv")
    TyyParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TyyParker.csv")
    TyzParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TyzParker.csv")
    TzzParker = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TzzParker.csv")

    TxxEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxxEGM96.csv")
    TxyEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxyEGM96.csv")
    TxzEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxzEGM96.csv")
    TyyEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TyyEGM96.csv")
    TyzEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TyzEGM96.csv")
    TzzEGM96 = load_data(
        "../../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TzzEGM96.csv")

    TxxTOTAL = TxxParker + TxxEGM96
    TxyTOTAL = TxyParker + TxyEGM96
    TxzTOTAL = TxzParker + TxzEGM96
    TyyTOTAL = TyyParker + TyyEGM96
    TyzTOTAL = TyzParker + TyzEGM96
    TzzTOTAL = TzzParker + TzzEGM96

    longitude_lowerbound_int = 50
    longitude_upperbound_int = 52
    latitude_lowerbound_int = 32
    latitude_upperbound_int = 34
    longitude_resolution = 0.5
    latitude_resolution = 0.5

    epsilon = 1e-9

    landa_range_deg = np.arange(
        longitude_lowerbound_int,
        longitude_upperbound_int + longitude_resolution + epsilon,
        longitude_resolution
    )
    landa_range_deg = np.round(landa_range_deg, 8)
    landa_range_deg = landa_range_deg[landa_range_deg <= longitude_upperbound_int]

    phi_range_deg = np.arange(
        latitude_lowerbound_int,
        latitude_upperbound_int + latitude_resolution + epsilon,
        latitude_resolution
    )
    phi_range_deg = np.round(phi_range_deg, 8)
    phi_range_deg = phi_range_deg[phi_range_deg <= latitude_upperbound_int]

    return (TxxParker, TxyParker, TxzParker,
            TyyParker, TyzParker, TzzParker,
            TxxEGM96, TxyEGM96, TxzEGM96,
            TyyEGM96, TyzEGM96, TzzEGM96,
            TxxTOTAL, TxyTOTAL, TxzTOTAL,
            TyyTOTAL, TyzTOTAL, TzzTOTAL,
            phi_range_deg, landa_range_deg
            )

# matrices = [TxxEGM96, TxyEGM96, TxzEGM96, TyyEGM96, TyzEGM96, TzzEGM96]
# titles = ["Txx EGM96", "Txy EGM96", "Txz EGM96", "Tyy EGM96", "Tyz EGM96", "Tzz EGM96"]
# start_time = time.time()
# plotter.plot_matrices(matrices, titles, phi_range_deg, landa_range_deg, None, start_time)
