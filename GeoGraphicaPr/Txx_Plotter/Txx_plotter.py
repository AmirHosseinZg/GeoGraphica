import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import functions
from tkinter import ttk


def on_closing():
    window.destroy()  # this will close the tkinter window
    exit()  # this will terminate the program


def plot_graph():
    try:
        # Define the range for landa and phi  , landa = longitude & phi = latitude
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

        # Create a meshgrid for plotting
        Phi, Landa = np.meshgrid(phi_range_deg, landa_range_deg)

        # Calculate the function
        r = float(radius.get())

        # Initialize the matrix to store Txx values
        Txx_values = np.zeros((len(landa_range), len(phi_range)))

        # Calculate Txx values
        # for i, landa in enumerate(landa_range):
        #     for j, phi in enumerate(phi_range):
        #         result = functions.Txx_function(r, phi, landa)
        #         print(f"landa = {landa},phi = {phi},Txx(landa,phi) = {result}")
        #         Txx_values[i, j] = result

        # input manually the data
        list1 = [-71.801965490127713434, -84.398795356459194127, -78.446212769524783884, -73.835622106115580208,
                 -72.700264980084424304]
        list2 = [-73.598232191375702387, -83.903631976890457738, -71.758302887084399903, -77.329656098854823543,
                 -76.224258015429363181]
        list3 = [-76.107635310711322735, -73.338602632907995252, -76.706483249978871592, -83.196567693392984932,
                 -71.176268395706854579]
        list4 = [-75.962068583923879351, -64.240101619566548434, -85.983387674377964636, -81.486942737915509376,
                 -64.348513862312606824]
        list5 = [-71.291576888951809049, -69.323676611527419178, -87.211709607239589741, -73.186455967003344677,
                 -67.249344827700329196]
        Txx_values[0] = list1
        Txx_values[1] = list2
        Txx_values[2] = list3
        Txx_values[3] = list4
        Txx_values[4] = list5
        # Plot the filled contour
        fig, ax = plt.subplots()

        # Create a filled contour plot
        contour_filled = ax.contourf(Landa, Phi, Txx_values, levels=int(contours.get()),
                                     cmap=selected_colormap.get())

        # Add contour lines on top
        contour_lines = ax.contour(Landa, Phi, Txx_values, levels=int(contours.get()), colors='black',
                                   linewidths=0.5)
        # Setting the division of the axes base on resolution
        ax.set_xticks(
            np.arange(longitude_lowerbound_int, longitude_upperbound_int + resolution_float, resolution_float))
        ax.set_yticks(np.arange(latitude_lowerbound_int, latitude_upperbound_int + resolution_float, resolution_float))

        # Add a colorbar to show the mapping of values to colors
        colorbar = fig.colorbar(contour_filled, ax=ax, label='Function Value')
        colorbar.set_ticks([float(Colorbar_lower_bound.get()), float(Colorbar_upper_bound.get())])

        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_title("Txx Function Plot")
        ax.grid(True)
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

# Create plot button
plot_button = ttk.Button(frame, text="Plot", command=plot_graph)
plot_button.grid(row=6, column=1, pady=5, columnspan=2)

# Create error label
error_label = ttk.Label(frame, text="", foreground="red")
error_label.grid(row=7, column=0, columnspan=4, pady=5)

# Create a list of colormaps available in matplotlib
colormaps = plt.colormaps()
# A variable to hold the selected color map
selected_colormap = tk.StringVar(window)
# set the default value for selected_colormap
selected_colormap.set(colormaps[0])
# Create a selection menu
ttk.OptionMenu(frame, selected_colormap, *colormaps).grid(row=6, column=0, padx=5, pady=5, sticky="ew")

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
