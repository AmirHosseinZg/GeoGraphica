# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the Excel file with all data (from the uploaded file)
# data_file_path = '../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.5/python/TxxTOTAL.csv'
# data = pd.read_csv(data_file_path, header=None)
#
# # Extract longitude, latitude, and Txx values from the file
# longitude = data.iloc[0, 1:].to_numpy()  # First row (excluding the first column)
# latitude = data.iloc[1:, 0].to_numpy()  # First column (excluding the first row)
# Txx_values = data.iloc[1:, 1:].to_numpy()  # The remaining grid of Txx values
#
# # Plot Txx vs Longitude for φ = 33°
# phi_fixed = 33  # Fixed latitude
# phi_index = np.argmin(np.abs(latitude - phi_fixed))  # Find the nearest latitude index
# Txx_vs_longitude = Txx_values[phi_index, :]  # Extract Txx values for the fixed latitude
#
# plt.figure()
# plt.plot(longitude, Txx_vs_longitude, label=f'Txx vs Longitude (φ={phi_fixed}°)')
# plt.xlabel('Longitude (degrees)')
# plt.ylabel('Txx (Eotvos)')
# plt.title(f'Txx vs Longitude for φ={phi_fixed}°')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Plot Txx vs Latitude for λ = 51°
# landa_fixed = 51  # Fixed longitude
# landa_index = np.argmin(np.abs(longitude - landa_fixed))  # Find the nearest longitude index
# Txx_vs_latitude = Txx_values[:, landa_index]  # Extract Txx values for the fixed longitude
#
# plt.figure()
# plt.plot(latitude, Txx_vs_latitude, label=f'Txx vs Latitude (λ={landa_fixed}°)')
# plt.xlabel('Latitude (degrees)')
# plt.ylabel('Txx (Eotvos)')
# plt.title(f'Txx vs Latitude for λ={landa_fixed}°')
# plt.legend()
# plt.grid()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file_1 = '../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.1/python/TxxTOTAL.csv'
data_file_2 = '../sources/final_phase/all_sources/project_result_csv_files/50-52,32-34,0.1/mathematica/MTxxTOTAL.csv'

data_1 = pd.read_csv(data_file_1, header=None)
data_2 = pd.read_csv(data_file_2, header=None)

longitude_1 = data_1.iloc[0, 1:].to_numpy()  # اولین سطر به عنوان طول جغرافیایی
latitude_1 = data_1.iloc[1:, 0].to_numpy()  # اولین ستون به عنوان عرض جغرافیایی
Txx_values_1 = data_1.iloc[1:, 1:].to_numpy()  # بقیه داده‌ها

longitude_2 = data_2.iloc[0, 1:].to_numpy()
latitude_2 = data_2.iloc[1:, 0].to_numpy()
Txx_values_2 = data_2.iloc[1:, 1:].to_numpy()

phi_fixed = 33
phi_index_1 = np.argmin(np.abs(latitude_1 - phi_fixed))
phi_index_2 = np.argmin(np.abs(latitude_2 - phi_fixed))

Txx_vs_longitude_1 = Txx_values_1[phi_index_1, :]
Txx_vs_longitude_2 = Txx_values_2[phi_index_2, :]

plt.figure()
plt.plot(longitude_1, Txx_vs_longitude_1, label=f'Python (φ={phi_fixed}°)', linestyle='-', color='b')
plt.plot(longitude_2, Txx_vs_longitude_2, label=f'Mathematica (φ={phi_fixed}°)', linestyle='--', color='r')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Txx (Eotvos)')
plt.title(f'Comparison of Txx vs Longitude for φ={phi_fixed}°')
plt.legend()
plt.grid()
plt.show()

landa_fixed = 51
landa_index_1 = np.argmin(np.abs(longitude_1 - landa_fixed))
landa_index_2 = np.argmin(np.abs(longitude_2 - landa_fixed))

Txx_vs_latitude_1 = Txx_values_1[:, landa_index_1]
Txx_vs_latitude_2 = Txx_values_2[:, landa_index_2]

plt.figure()
plt.plot(latitude_1, Txx_vs_latitude_1, label=f'Python (λ={landa_fixed}°)', linestyle='-', color='b')
plt.plot(latitude_2, Txx_vs_latitude_2, label=f'Mathematica (λ={landa_fixed}°)', linestyle='--', color='r')
plt.xlabel('Latitude (degrees)')
plt.ylabel('Txx (Eotvos)')
plt.title(f'Comparison of Txx vs Latitude for λ={landa_fixed}°')
plt.legend()
plt.grid()
plt.show()
