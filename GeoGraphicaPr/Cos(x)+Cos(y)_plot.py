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

        # Plot the contour
        fig, ax = plt.subplots()
        ax.contour(X, Y, Z, levels=int(contours.get()), cmap='viridis')  # levels define the number of contours
        # Add a colorbar
        ax.plot(x, y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(
            f' ContourPlot [Cos(x) + Cos(y) , (x,{x_lower_bound.get()},{x_upper_bound.get()}) , (y,{y_lower_bound.get()},{y_upper_bound.get()}) , contours -> 10 ]')
        ax.grid(True)

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
