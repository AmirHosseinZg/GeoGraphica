import numpy as np
import functions
import matplotlib.pyplot as plt

# Define the range and resolution in degrees
phi_range_deg = np.arange(32, 34.5, 0.5)  # 32 to 34 with step 0.5
landa_range_deg = np.arange(50, 52.5, 0.5)  # 50 to 52 with step 0.5

# Convert degrees to radians
phi_range = np.radians(phi_range_deg)
landa_range = np.radians(landa_range_deg)

# Create a meshgrid for plotting
Phi, Landa = np.meshgrid(phi_range_deg, landa_range_deg)


# Initialize the matrix to store Txx values
Txx_values = np.zeros((len(phi_range), len(landa_range)))

# Calculate Txx values
for i, phi in enumerate(phi_range):
    for j, landa in enumerate(landa_range):
        Txx_values[i, j] = functions.Txx_function(6730000, phi, landa)


# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(Landa, Phi, Txx_values, cmap='viridis', levels=100)
plt.colorbar(label='Txx value')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Txx Function Plot')
plt.show()
