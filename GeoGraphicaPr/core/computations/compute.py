from GeoGraphicaPr.core.computations import Constant
from GeoGraphicaPr.core.computations.functions import Txx_function, Txy_function, Txz_function, Tyy_function, \
    Tyz_function, Tzz_function
from GeoGraphicaPr.utils import files_paths
from joblib import Parallel, delayed
from mpmath import mp, factorial
import numpy as np
import pandas as pd
import warnings
import os

# Constants for calculations
constants = Constant.Constants()

EOTVOS = constants.EOTVOS
A = (constants.get_a())
h = (constants.get_h())
e2 = (constants.get_e2())
G = (constants.get_G())
p = (constants.get_p())


def compute_gradients(longitude_lowerbound_int, longitude_upperbound_int,
                      latitude_lowerbound_int, latitude_upperbound_int,
                      longitude_resolution, latitude_resolution):
    # Define the range and resolution in degrees

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

    print(f"{longitude_lowerbound_int=}\n{longitude_upperbound_int=}")
    print(f"{latitude_lowerbound_int=}\n{latitude_upperbound_int=}")
    print(f"{longitude_resolution=}\n{latitude_resolution=}")
    print(f"{landa_range_deg=}\n{phi_range_deg=}")

    # Suppress UserWarnings
    warnings.simplefilter(action='ignore', category=UserWarning)

    # Load the Excel file
    Elev_Matrix_Output_file_path = files_paths.Elev_Matrix_Output_file_path
    try:
        data = pd.ExcelFile(Elev_Matrix_Output_file_path)
        # Parse the first sheet into a DataFrame without headers or index
        df = pd.read_excel(Elev_Matrix_Output_file_path, header=None, index_col=None)
        # Convert the DataFrame into a 2D list (matrix)
        elev_data = df.values.tolist()
        print("Elev data matrix successfully loaded.")
    except FileNotFoundError:
        print(f"Error: The file at {Elev_Matrix_Output_file_path} was not found.")
        elev_data = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        elev_data = None

    # Create a function to access elements with 1-based indexing
    def get_element(matrix, row, col):
        try:
            if matrix is None:
                raise ValueError("The matrix data is not loaded.")
            return matrix[row - 1][col - 1]
        except IndexError:
            print("Error: The specified row or column is out of bounds.")
        except ValueError as ve:
            print(f"Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Define functions for grid indexing
    def i(phi):
        return np.round(1 + (phi + 10.0042) * 60 / 5).astype(int)

    def j(lam):
        return np.round(1 + (lam - 20.0042) * 60 / 5).astype(int)

    # Example: Access the first element (equivalent to A1 in Excel)
    i_index = i(phi_range_deg)
    j_index = j(landa_range_deg)
    Z_list = []
    try:
        for i in i_index:
            tmp = []
            for j in j_index:
                result = get_element(elev_data, i, j)
                tmp.append(result)
            Z_list.append(tmp)
    except Exception as e:
        print(f"An unexpected error occurred while accessing the matrix: {e}")

    Z = np.array(Z_list)

    # Calculate grid indices
    imin = min(i_index)
    imax = max(i_index)
    jmin = min(j_index)
    jmax = max(j_index)

    # Calculate mean elevation
    mean_elev = 0
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            result = get_element(elev_data, i, j)
            mean_elev += result
    data_nums = (imax - imin + 1) * (jmax - jmin + 1)
    mean_elev = mean_elev / data_nums

    # Final altitude
    alt = h + mean_elev

    # Middle point calculations
    phi_middle = ((latitude_lowerbound_int + latitude_upperbound_int) / 2) * np.pi / 180
    phi_middle2 = np.arctan((1 - float(e2)) * np.tan(phi_middle))
    landa_middle = ((longitude_lowerbound_int + longitude_upperbound_int) / 2) * np.pi / 180
    v_middle = A / np.sqrt(1 - e2 * (np.sin(phi_middle) ** 2))

    # Radius at the middle point
    X_ECEF_middle = v_middle * np.cos(phi_middle) * np.cos(landa_middle)
    Y_ECEF_middle = v_middle * np.cos(phi_middle) * np.sin(landa_middle)
    Z_ECEF_middle = v_middle * (1 - e2) * np.sin(phi_middle)
    R_middle = np.sqrt(X_ECEF_middle ** 2 + Y_ECEF_middle ** 2 + Z_ECEF_middle ** 2)

    # X and Y distances
    X_dist = R_middle * np.cos(phi_middle2) * (longitude_upperbound_int - longitude_lowerbound_int) * np.pi / 180
    Y_dist = R_middle * (np.arctan((1 - float(e2)) * np.tan(latitude_upperbound_int * np.pi / 180)) -
                         np.arctan((1 - float(e2)) * np.tan(latitude_lowerbound_int * np.pi / 180)))

    # X and Y intervals
    Xint = R_middle * np.cos(phi_middle2) * (longitude_resolution * np.pi / 180)
    Yint = R_middle * np.arctan((1 - float(e2)) * np.tan(latitude_resolution * np.pi / 180))

    # Calculate the number of points needed to match Mathematica's approach
    num_points_x = len(landa_range_deg)
    num_points_y = len(phi_range_deg)

    # Use np.linspace() to get exactly num_points_x points
    Xlim = np.linspace(0, X_dist, num_points_x)
    Ylim = np.linspace(0, Y_dist, num_points_y)

    # Meshgrid
    X_mesh, Y_mesh = np.meshgrid(Xlim, Ylim)

    # Calculate grid size and grid points
    M1 = num_points_x  # Number of points along x
    M2 = num_points_y  # Number of points along y
    dx = Xint
    dy = Yint

    # Generate indices for p1 and p2
    p1 = np.arange(0, M1)
    p2 = np.arange(0, M2)

    # Initialize fp1 and fp2 arrays
    fp1 = np.zeros(M1)
    fp2 = np.zeros(M2)

    # Populate fp1 array
    for i in range(M1):
        if p1[i] <= M1 // 2 - 1:
            fp1[i] = p1[i] / (dx * M1)
        else:
            fp1[i] = (p1[i] - M1) / (dx * M1)

    # Populate fp2 array
    for j in range(M2):
        if p2[j] <= M2 // 2 - 1:
            fp2[j] = p2[j] / (dy * M2)
        else:
            fp2[j] = (p2[j] - M2) / (dy * M2)

    # Create fp1mesh and fp2mesh
    fp1mesh, fp2mesh = np.meshgrid(fp1, fp2, indexing='ij')
    fp1mesh, fp2mesh = fp1mesh.T, fp2mesh.T

    # Compute the final fp matrix
    fp = np.sqrt(fp1mesh ** 2 + fp2mesh ** 2)
    fp[0, 0] = 1e-17  # Avoid division by zero

    # Calculate sig using Fourier series expansion
    sig = np.zeros((M1, M2), dtype=object)

    # Loop to calculate the Fourier series terms

    # Use mpmath equivalents for operations involving mpf values
    two_pi_fp = 2 * mp.pi * fp

    for n in range(1, 21):
        # Calculate the factorial using mpmath (keeps high precision)
        fact_n = factorial(n)
        # Calculate term using mpf and mpmath functions
        Z_powered = np.power(Z, n)
        term_fft = np.fft.fft2(Z_powered)
        term_fft = 1 / len(Z) * term_fft
        term_fft = np.conjugate(term_fft)
        term = (1 / fact_n) * (two_pi_fp ** (n - 2)) * term_fft
        # Add the calculated term to sig
        for i in range(M1):
            for j in range(M2):
                sig[i][j] += term[i][j]

    # Define the imaginary unit as i
    i = 1j

    miu_xx = (-1) * pow((2 * np.pi), 2) * pow(fp1mesh, 2)
    miu_xy = (-1) * pow((2 * np.pi), 2) * fp1mesh * fp2mesh
    miu_xz = (-i) * pow((2 * np.pi), 2) * fp1mesh * fp
    miu_yy = (-1) * pow((2 * np.pi), 2) * pow(fp2mesh, 2)
    miu_yz = (-i) * pow((2 * np.pi), 2) * fp2mesh * fp
    miu_zz = pow((2 * np.pi), 2) * pow(fp, 2)

    alt_float = float(alt)
    fp_float = np.array(fp, dtype=float)

    def inverse_fourier_mathematica(matrix_, scaling_factor=1 / len(Z)):
        matrix_ = np.array(matrix_, dtype=np.complex128)
        matrix_conjugate = np.conjugate(matrix_)
        inverse_fft = np.fft.ifft2(matrix_conjugate)
        result = scaling_factor * inverse_fft * np.prod(matrix_.shape)
        return result

    TxxParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_xx * np.exp(-2 * np.pi * alt_float * fp_float) * sig)
    TxyParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_xy * np.exp(-2 * np.pi * alt_float * fp_float) * sig)
    TxzParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_xz * np.exp(-2 * np.pi * alt_float * fp_float) * sig)
    TyyParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_yy * np.exp(-2 * np.pi * alt_float * fp_float) * sig)
    TyzParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_yz * np.exp(-2 * np.pi * alt_float * fp_float) * sig)
    TzzParker = (2 * np.pi) * G * p * inverse_fourier_mathematica(
        miu_zz * np.exp(-2 * np.pi * alt_float * fp_float) * sig)

    TxxParker = np.real(TxxParker) / EOTVOS
    TxyParker = np.real(TxyParker) / EOTVOS
    TxzParker = np.real(TxzParker) / EOTVOS
    TyyParker = np.real(TyyParker) / EOTVOS
    TyzParker = np.real(TyzParker) / EOTVOS
    TzzParker = np.real(TzzParker) / EOTVOS

    # Define the base directory for CSV output files
    base_path = files_paths.base_path

    # Ensure the directory exists (creates it if it doesn't)
    os.makedirs(base_path, exist_ok=True)

    # Export individual Parker matrices to CSV
    def export_matrix(filename, matrix):
        # Define the full path
        filepath = os.path.join(base_path, filename)

        # Prepare the data with the additional row and column for latitude and longitude
        data = np.zeros((M1 + 1, M2 + 1))
        data[1:, 1:] = matrix
        data[0, 1:] = landa_range_deg
        data[1:, 0] = phi_range_deg

        # Save to CSV
        pd.DataFrame(data).to_csv(filepath, index=False, header=False)

    # Export each matrix

    TxxParker_file_name = files_paths.TxxParker_file_name
    TxyParker_file_name = files_paths.TxyParker_file_name
    TxzParker_file_name = files_paths.TxzParker_file_name
    TyyParker_file_name = files_paths.TyyParker_file_name
    TyzParker_file_name = files_paths.TyzParker_file_name
    TzzParker_file_name = files_paths.TzzParker_file_name

    export_matrix(TxxParker_file_name, TxxParker)
    export_matrix(TxyParker_file_name, TxyParker)
    export_matrix(TxzParker_file_name, TxzParker)
    export_matrix(TyyParker_file_name, TyyParker)
    export_matrix(TyzParker_file_name, TyzParker)
    export_matrix(TzzParker_file_name, TzzParker)

    # Combine all matrices into allTparker
    allTparker = np.zeros((M1 * M2 + 1, 8), dtype=object)
    allTparker[0] = ["Latitude(deg)", "Longitude(deg)", "TxxParker", "TxyParker", "TxzParker", "TyyParker",
                     "TyzParker",
                     "TzzParker"]
    idx = 1
    for i, lat in enumerate(phi_range_deg):
        for j, lon in enumerate(landa_range_deg):
            allTparker[idx] = [lat, lon, TxxParker[i, j], TxyParker[i, j], TxzParker[i, j], TyyParker[i, j],
                               TyzParker[i, j], TzzParker[i, j]]
            idx += 1

    # Export combined matrix
    allTparker_file_name = files_paths.allTparker_file_name
    combined_filepath = os.path.join(base_path, allTparker_file_name)
    pd.DataFrame(allTparker).to_csv(combined_filepath, index=False, header=False)

    alt = h + mean_elev

    # Convert latitude and longitude ranges to radians
    phi_lim_rad = np.radians(phi_range_deg)
    lambda_lim_rad = np.radians(landa_range_deg)

    # Calculate geocentric latitude
    phi_lim_centric = np.arctan((1 - e2) * np.tan(phi_lim_rad))
    lambda_lim_centric = lambda_lim_rad

    # Create mesh grid for phi and lambda
    phi_mesh = np.tile(phi_lim_centric, (len(lambda_lim_centric), 1)).T
    lambda_mesh = np.tile(lambda_lim_centric, (len(phi_lim_centric), 1))
    v = A / np.sqrt(1 - e2 * np.sin(phi_lim_rad) ** 2)

    # Calculate ECEF coordinates
    XECEF = np.outer(v + alt, np.cos(phi_lim_rad)) * np.cos(lambda_lim_rad)
    YECEF = np.outer(v + alt, np.cos(phi_lim_rad)) * np.sin(lambda_lim_rad)
    ZECEF = (v * (1 - e2) + alt) * np.sin(phi_lim_rad)

    # Calculate radius r
    r = np.sqrt(XECEF ** 2 + YECEF ** 2 + ZECEF ** 2)

    def compute_row_generic(i, r_row, phi_row, lambda_row, maximum_of_counter, function):
        """
        Generic function to compute a row for a specific tensor.

        Parameters:
        - i: Current row index.
        - r_row: Row of radii.
        - phi_row: Row of latitudes in radians.
        - lambda_row: Row of longitudes in radians.
        - maximum_of_counter: Maximum counter value.
        - function: The function to compute values (Txx_function, Txy_function, etc.).

        Returns:
        - Computed row.
        """
        return [function(r_row[j], phi_row[j], lambda_row[j], maximum_of_counter) for j in range(len(r_row))]

    # Calculate maximum_of_counter based on resolution
    maximum_of_counter = int(
        ((latitude_upperbound_int - latitude_lowerbound_int) / latitude_resolution) + 1
    )

    # Parallel computation for each tensor
    TxxEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Txx_function)
        for i in range(phi_mesh.shape[0])
    )
    TxxEGM96 = np.array(TxxEGM96)

    TxyEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Txy_function)
        for i in range(phi_mesh.shape[0])
    )
    TxyEGM96 = np.array(TxyEGM96)

    TxzEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Txz_function)
        for i in range(phi_mesh.shape[0])
    )
    TxzEGM96 = np.array(TxzEGM96)

    TyyEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Tyy_function)
        for i in range(phi_mesh.shape[0])
    )
    TyyEGM96 = np.array(TyyEGM96)

    TyzEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Tyz_function)
        for i in range(phi_mesh.shape[0])
    )
    TyzEGM96 = np.array(TyzEGM96)

    TzzEGM96 = Parallel(n_jobs=-1)(
        delayed(compute_row_generic)(i, r[i], phi_mesh[i], lambda_mesh[i], maximum_of_counter, Tzz_function)
        for i in range(phi_mesh.shape[0])
    )
    TzzEGM96 = np.array(TzzEGM96)

    # Export each EGM96 matrix to CSV
    def export_matrix(filename, matrix):
        data = np.zeros((len(phi_range_deg) + 1, len(landa_range_deg) + 1))
        data[1:, 1:] = matrix
        data[0, 1:] = landa_range_deg
        data[1:, 0] = phi_range_deg
        pd.DataFrame(data).to_csv(os.path.join(base_path, filename), index=False, header=False)

    TxxEGM96_file_name = files_paths.TxxEGM96_file_name
    TxyEGM96_file_name = files_paths.TxyEGM96_file_name
    TxzEGM96_file_name = files_paths.TxzEGM96_file_name
    TyyEGM96_file_name = files_paths.TyyEGM96_file_name
    TyzEGM96_file_name = files_paths.TyzEGM96_file_name
    TzzEGM96_file_name = files_paths.TzzEGM96_file_name
    allTEGM96_file_name = files_paths.allTEGM96_file_name

    export_matrix(TxxEGM96_file_name, TxxEGM96)
    export_matrix(TxyEGM96_file_name, TxyEGM96)
    export_matrix(TxzEGM96_file_name, TxzEGM96)
    export_matrix(TyyEGM96_file_name, TyyEGM96)
    export_matrix(TyzEGM96_file_name, TyzEGM96)
    export_matrix(TzzEGM96_file_name, TzzEGM96)

    # Combine all matrices into a single output CSV
    allTegm96 = np.zeros((len(phi_range_deg) * len(landa_range_deg) + 1, 8), dtype=object)
    allTegm96[0] = ["Latitude(deg)", "Longitude(deg)", "TxxEGM96", "TxyEGM96", "TxzEGM96", "TyyEGM96", "TyzEGM96",
                    "TzzEGM96"]

    idx = 1
    for i, lat in enumerate(phi_range_deg):
        for j, lon in enumerate(landa_range_deg):
            allTegm96[idx] = [lat, lon, TxxEGM96[i, j], TxyEGM96[i, j], TxzEGM96[i, j], TyyEGM96[i, j],
                              TyzEGM96[i, j],
                              TzzEGM96[i, j]]
            idx += 1

    # Export the combined matrix
    pd.DataFrame(allTegm96).to_csv(os.path.join(base_path, allTEGM96_file_name), index=False, header=False)

    # Compute total matrices
    TxxTOTAL = TxxParker + TxxEGM96
    TxyTOTAL = TxyParker + TxyEGM96
    TxzTOTAL = TxzParker + TxzEGM96
    TyyTOTAL = TyyParker + TyyEGM96
    TyzTOTAL = TyzParker + TyzEGM96
    TzzTOTAL = TzzParker + TzzEGM96

    # Define function to export individual matrices to CSV with headers
    def export_matrix(filename, matrix):
        data = np.zeros((len(phi_range_deg) + 1, len(landa_range_deg) + 1))
        data[1:, 1:] = matrix
        data[0, 1:] = landa_range_deg
        data[1:, 0] = phi_range_deg
        pd.DataFrame(data).to_csv(os.path.join(base_path, filename), index=False, header=False)

    # Export each total matrix to CSV

    TxxTOTAL_file_name = files_paths.TxxTOTAL_file_name
    TxyTOTAL_file_name = files_paths.TxyTOTAL_file_name
    TxzTOTAL_file_name = files_paths.TxzTOTAL_file_name
    TyyTOTAL_file_name = files_paths.TyyTOTAL_file_name
    TyzTOTAL_file_name = files_paths.TyzTOTAL_file_name
    TzzTOTAL_file_name = files_paths.TzzTOTAL_file_name
    allTTOTAL_file_name = files_paths.allTTOTAL_file_name

    export_matrix(TxxTOTAL_file_name, TxxTOTAL)
    export_matrix(TxyTOTAL_file_name, TxyTOTAL)
    export_matrix(TxzTOTAL_file_name, TxzTOTAL)
    export_matrix(TyyTOTAL_file_name, TyyTOTAL)
    export_matrix(TyzTOTAL_file_name, TyzTOTAL)
    export_matrix(TzzTOTAL_file_name, TzzTOTAL)

    # Combine all matrices into a single CSV file
    allTtotal = np.zeros((len(phi_range_deg) * len(landa_range_deg) + 1, 8), dtype=object)
    allTtotal[0] = ["Latitude(deg)", "Longitude(deg)", "Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz"]

    # Fill in the combined matrix
    idx = 1
    for i, lat in enumerate(phi_range_deg):
        for j, lon in enumerate(landa_range_deg):
            allTtotal[idx] = [lat, lon, TxxTOTAL[i, j], TxyTOTAL[i, j], TxzTOTAL[i, j],
                              TyyTOTAL[i, j], TyzTOTAL[i, j], TzzTOTAL[i, j]]
            idx += 1

    # Export the combined matrix
    pd.DataFrame(allTtotal).to_csv(os.path.join(base_path, allTTOTAL_file_name), index=False, header=False)

    return (TxxParker, TxyParker, TxzParker,
            TyyParker, TyzParker, TzzParker,
            TxxEGM96, TxyEGM96, TxzEGM96,
            TyyEGM96, TyzEGM96, TzzEGM96,
            TxxTOTAL, TxyTOTAL, TxzTOTAL,
            TyyTOTAL, TyzTOTAL, TzzTOTAL,
            phi_range_deg, landa_range_deg
            )
