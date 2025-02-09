import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file with all data (from the uploaded file)
data_file_path = '../sources/final_phase/all_sources/project_result_csv_files/txxTOTAL.csv'
data = pd.read_csv(data_file_path, header=None)

# Extract longitude, latitude, and Txx values from the file
longitude = data.iloc[0, 1:].to_numpy()  # First row (excluding the first column)
latitude = data.iloc[1:, 0].to_numpy()  # First column (excluding the first row)
Txx_values = data.iloc[1:, 1:].to_numpy()  # The remaining grid of Txx values

# Plot Txx vs Longitude for φ = 33°
phi_fixed = 33  # Fixed latitude
phi_index = np.argmin(np.abs(latitude - phi_fixed))  # Find the nearest latitude index
Txx_vs_longitude = Txx_values[phi_index, :]  # Extract Txx values for the fixed latitude

plt.figure()
plt.plot(longitude, Txx_vs_longitude, label=f'Txx vs Longitude (φ={phi_fixed}°)')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Txx (Eotvos)')
plt.title(f'Txx vs Longitude for φ={phi_fixed}°')
plt.legend()
plt.grid()
plt.show()

# Plot Txx vs Latitude for λ = 51°
landa_fixed = 51  # Fixed longitude
landa_index = np.argmin(np.abs(longitude - landa_fixed))  # Find the nearest longitude index
Txx_vs_latitude = Txx_values[:, landa_index]  # Extract Txx values for the fixed longitude

plt.figure()
plt.plot(latitude, Txx_vs_latitude, label=f'Txx vs Latitude (λ={landa_fixed}°)')
plt.xlabel('Latitude (degrees)')
plt.ylabel('Txx (Eotvos)')
plt.title(f'Txx vs Latitude for λ={landa_fixed}°')
plt.legend()
plt.grid()
plt.show()
