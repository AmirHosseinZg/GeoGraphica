import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions
from tkinter import ttk
from scipy.interpolate import griddata  # interpolation
import time


def on_closing():
    window.destroy()  # this will close the tkinter window
    exit()  # this will terminate the program


def plot_graph():
    try:
        # Start time to measure execution time
        start_time = time.time()

        # Define the range for landa and phi (longitude & latitude)
        longitude_lowerbound_int = int(longitude_lowerbound.get())
        longitude_upperbound_int = int(longitude_upperbound.get())
        latitude_lowerbound_int = int(latitude_lowerbound.get())
        latitude_upperbound_int = int(latitude_upperbound.get())

        resolution_float = float(resolution.get())

        # Define the range and resolution in degrees
        landa_range_deg = np.arange(longitude_lowerbound_int, longitude_upperbound_int + resolution_float,
                                    resolution_float)  # example : 50 to 52 with step 0.5
        phi_range_deg = np.arange(latitude_lowerbound_int, latitude_upperbound_int + resolution_float,
                                  resolution_float)  # example : 32 to 34 with step 0.5

        # Convert degrees to radians
        landa_range = np.radians(landa_range_deg)
        phi_range = np.radians(phi_range_deg)

        # Increase resolution of the mesh for smoother curves
        high_res_landa = np.linspace(longitude_lowerbound_int, longitude_upperbound_int, 100)
        high_res_phi = np.linspace(latitude_lowerbound_int, latitude_upperbound_int, 100)
        high_res_Landa, high_res_Phi = np.meshgrid(high_res_landa, high_res_phi)

        # Radius
        r = float(radius.get())

        # Initialize the matrix to store Txx values
        # len_landa_range = int(float((longitude_upperbound_int - longitude_lowerbound_int))/resolution_float + 1)
        # len_phi_range = int(float((latitude_upperbound_int - latitude_lowerbound_int))/resolution_float + 1)
        Txx_values = np.zeros((len(landa_range), len(phi_range)))

        # Calculate Txx values
        # for i, landa in enumerate(landa_range):
        #     for j, phi in enumerate(phi_range):
        #         result = functions.Txx_function(r, phi, landa)
        #         print(f"landa = {landa}, phi = {phi}, Txx(landa,phi) = {result}")
        #         Txx_values[i, j] = result

        # input manually the data for res0.5
        # list1 = [-71.801965490127713434, -84.398795356459194127, -78.446212769524783884, -73.835622106115580208,
        #          -72.700264980084424304]
        # list2 = [-73.598232191375702387, -83.903631976890457738, -71.758302887084399903, -77.329656098854823543,
        #          -76.224258015429363181]
        # list3 = [-76.107635310711322735, -73.338602632907995252, -76.706483249978871592, -83.196567693392984932,
        #          -71.176268395706854579]
        # list4 = [-75.962068583923879351, -64.240101619566548434, -85.983387674377964636, -81.486942737915509376,
        #          -64.348513862312606824]
        # list5 = [-71.291576888951809049, -69.323676611527419178, -87.211709607239589741, -73.186455967003344677,
        #          -67.249344827700329196]
        # Txx_values[0] = list1
        # Txx_values[1] = list2
        # Txx_values[2] = list3
        # Txx_values[3] = list4
        # Txx_values[4] = list5

        # input manually the data for res0.1
        file_path = \
            "D:\\programming\Projects\\GeoGraphica\\Sources\\Txx_OutPuts_example\\res0.1\\res0.1_data_outputs.xlsx"
        df = pd.read_excel(file_path)

        for index, row in df.iterrows():
            row_list = row.tolist()
            Txx_values[index] = row_list

        # Interpolating the Txx values for higher resolution
        points = np.array([(landa, phi) for landa in landa_range_deg for phi in phi_range_deg])
        values = Txx_values.flatten()
        grid_z = griddata(points, values, (high_res_Landa, high_res_Phi), method='cubic')

        # Plot the filled contour
        fig, ax = plt.subplots()

        # Create a filled contour plot with selected colormap
        contour_filled = ax.contourf(high_res_Landa, high_res_Phi, grid_z, levels=int(contours.get()),
                                     cmap=selected_colormap.get())

        # Add contour lines on top
        contour_lines = ax.contour(high_res_Landa, high_res_Phi, grid_z, levels=int(contours.get()), colors='black',
                                   linewidths=0.5)

        # Colorbar limits based on user input or automatic
        if auto_colorbar.get() == 1:  # Automatic mode
            vmin, vmax = None, None  # Automatic color range
        else:  # Manual mode
            vmin = float(Colorbar_lower_bound.get())
            vmax = float(Colorbar_upper_bound.get())

        # Add a colorbar with manual or automatic range
        colorbar = fig.colorbar(contour_filled, ax=ax, label='Function Value')

        # Handle manual or automatic colorbar ranges
        if vmin is not None and vmax is not None:  # Manual mode
            ticks = np.linspace(vmin, vmax, 5)  # Ensure at least 5 ticks
            if vmin <= 0 <= vmax and 0 not in ticks:  # Only add 0 if it's within the range
                ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
            colorbar.set_ticks(ticks)
        else:  # Automatic mode
            ticks = colorbar.get_ticks()  # Get automatic ticks
            if ticks[0] <= 0 <= ticks[-1] and 0 not in ticks:  # Ensure 0 is within the range
                ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
            colorbar.set_ticks(ticks)

        # Set the colorbar label
        colorbar.set_label('Function Value')

        # Setting the division of the axes based on resolution
        ax.set_xticks(
            np.arange(longitude_lowerbound_int, longitude_upperbound_int + resolution_float, resolution_float))
        ax.set_yticks(np.arange(latitude_lowerbound_int, latitude_upperbound_int + resolution_float, resolution_float))

        # Labels and title
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_title("Txx Function Plot")
        ax.grid(True)

        # End time and calculate execution duration
        end_time = time.time()
        total_spent_time = end_time - start_time
        print(functions.convert_seconds(total_spent_time))

        # Show the plot
        plt.show()

    except ValueError:
        error_label.config(text="Please enter valid numeric values.")


# create the main window
window = tk.Tk()
window.title("Txx Plotter")

# create the input frame
frame = ttk.Frame(window, padding=100)
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter the range of longitude changes from : ").grid(row=0, column=0, padx=5, pady=5)
longitude_lowerbound = ttk.Entry(frame)
longitude_lowerbound.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text=" To ").grid(row=0, column=2, padx=5, pady=5)
longitude_upperbound = ttk.Entry(frame)
longitude_upperbound.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="Enter the range of Latitude changes from : ").grid(row=1, column=0, padx=5, pady=5)
latitude_lowerbound = ttk.Entry(frame)
latitude_lowerbound.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text=" To ").grid(row=1, column=2, padx=5, pady=5)
latitude_upperbound = ttk.Entry(frame)
latitude_upperbound.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="Enter your desired radius : ").grid(row=2, column=0, padx=5, pady=5)
radius = ttk.Entry(frame)
radius.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="Enter your desired resolution : ").grid(row=3, column=0, padx=5, pady=5)
resolution = ttk.Entry(frame)
resolution.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="Enter number of contours :          ").grid(row=4, column=0, padx=5, pady=5)
contours = ttk.Entry(frame)
contours.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="Enter Colorbar range of changes :  from ").grid(row=5, column=0, padx=5, pady=5)
Colorbar_lower_bound = ttk.Entry(frame)
Colorbar_lower_bound.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

ttk.Label(frame, text="to ").grid(row=5, column=2, padx=5, pady=5)
Colorbar_upper_bound = ttk.Entry(frame)
Colorbar_upper_bound.grid(row=5, column=3, padx=5, pady=5, sticky="ew")

# Checkbox for automatic colorbar
auto_colorbar = tk.IntVar()
ttk.Checkbutton(frame, text="Automatic Colorbar", variable=auto_colorbar).grid(row=6, column=0, padx=5, pady=5)

# Create plot button
plot_button = ttk.Button(frame, text="Plot", command=plot_graph)
plot_button.grid(row=7, column=1, pady=5, columnspan=2)

# Create error label
error_label = ttk.Label(frame, text="", foreground="red")
error_label.grid(row=8, column=0, columnspan=4, pady=5)

# Create a list of colormaps available in matplotlib
# Retrieve available colormaps from matplotlib
colormaps = plt.colormaps()

# A variable to hold the selected colormap
selected_colormap = tk.StringVar(window)

# Create a selection menu for colormaps(Set the default value for selected_colormap to a blue-red colormap)
colormap_menu = ttk.OptionMenu(frame, selected_colormap, 'RdBu', *colormaps)
colormap_menu.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

# Configure rows and columns to be resizable
frame.grid_rowconfigure(0, weight=1)
frame.grid_rowconfigure(1, weight=1)
frame.grid_rowconfigure(2, weight=1)
frame.grid_rowconfigure(3, weight=1)
frame.grid_rowconfigure(4, weight=1)
frame.grid_rowconfigure(5, weight=1)
frame.grid_rowconfigure(6, weight=1)

frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)
frame.grid_columnconfigure(2, weight=1)
frame.grid_columnconfigure(3, weight=1)

window.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
window.mainloop()
