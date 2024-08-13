import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def on_closing():
    window.destroy()  # this will close the tkinter window
    exit()  # this will terminate the program


def plot_graph():
    try:
        # Define the range for x and y
        x = np.linspace(int(x_lower_bound.get()), int(x_upper_bound.get()), 1000)
        y = np.linspace(int(y_lower_bound.get()), int(y_upper_bound.get()), 1000)
        X, Y = np.meshgrid(x, y)

        # Calculate the function
        Z = np.cos(X) + np.cos(Y)

        # Plot the filled contour
        fig, ax = plt.subplots()
        """plt.subplot() return an figure object and an axis object . fig represents the overall shape and ax represents
        the area in which the graph is drawn """

        # The following line of code creates a filled contour plot
        contour_filled = ax.contourf(X, Y, Z, levels=int(contours.get()), cmap='viridis')
        """cmap='viridis': the colormap used to fill the areas between the contour lines. viridis is one of matplotlib's
         default options, which has a yellow-green-blue color spectrum."""

        # Add contour lines on top
        contour_lines = ax.contour(X, Y, Z, levels=int(contours.get()), colors='black', linewidths=0.5)
        """colors='black': makes the color of contours lines black 
        linewidths=0.5: specifies the thickness of the contour lines , which here is equal to 0.5 units
        Output: This function creates a QuadContourSet object named contour_lines that represents the contour lines."""

        # Add a colorbar to show the mapping of values to colors
        fig.colorbar(contour_filled, ax=ax, label='Function Value')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(
            f' ContourPlot [Cos(x) + Cos(y) , (x,{x_lower_bound.get()},{x_upper_bound.get()}) , (y,{y_lower_bound.get()},{y_upper_bound.get()}) , contours -> {contours.get()} ]')
        ax.grid(True)

        # Display the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except ValueError:
        error_label.config(text="Please enter valid numeric values.")


# Create the main window
window = tk.Tk()
window.title("Cos(x) + Cos(y) Plotter")

# Create input fields
frame = ttk.Frame(window, padding="100")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter x range of changes :  from").grid(row=0, column=0, padx=5, pady=5)
x_lower_bound = ttk.Entry(frame)
x_lower_bound.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame, text="to ").grid(row=0, column=2, padx=5, pady=5)
x_upper_bound = ttk.Entry(frame)
x_upper_bound.grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame, text="Enter y range of changes :  from ").grid(row=1, column=0, padx=5, pady=5)
y_lower_bound = ttk.Entry(frame)
y_lower_bound.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame, text="to ").grid(row=1, column=2, padx=5, pady=5)
y_upper_bound = ttk.Entry(frame)
y_upper_bound.grid(row=1, column=3, padx=5, pady=5)

ttk.Label(frame, text="Enter number of contours :          ").grid(row=2, column=0, padx=5, pady=5)
contours = ttk.Entry(frame)
contours.grid(row=2, column=1, padx=5, pady=5)

# Create plot button
plot_button = ttk.Button(frame, text="Plot", command=plot_graph)
plot_button.grid(row=3, column=0, pady=5, columnspan=2)

# Create error label
error_label = ttk.Label(frame, text="", foreground="red")
error_label.grid(row=4, column=0, columnspan=2, pady=5)

window.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
window.mainloop()
