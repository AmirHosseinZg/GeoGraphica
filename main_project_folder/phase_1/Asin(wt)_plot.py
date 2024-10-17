import tkinter as tk  # For creating the GUI
from tkinter import ttk  # For themed widgets
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # To embed matplotlib in Tkinter


def plot_graph():
    try:
        # Get user input and convert to appropriate types
        A = float(entry_A.get())
        omega = float(entry_omega.get())
        periods = int(entry_periods.get())

        # Generate time values and compute the sine wave
        t = np.linspace(0, 2 * np.pi * periods, 1000)
        y = A * np.sin(omega * t)

        # Create a new figure and plot the sine wave
        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'A sin(ωt) with A={A}, ω={omega}, and {periods} periods')
        ax.grid(True)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except ValueError:
        # Display error message if input is invalid
        error_label.config(text="Please enter valid numeric values.")


# Create the main window
window = tk.Tk()
window.title("Sinusoidal Plotter")

# Create input fields for A, ω, and periods
frame = ttk.Frame(window, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter A:").grid(row=0, column=0, padx=5, pady=5)
entry_A = ttk.Entry(frame)
entry_A.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame, text="Enter ω:").grid(row=1, column=0, padx=5, pady=5)
entry_omega = ttk.Entry(frame)
entry_omega.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame, text="Enter periods:").grid(row=2, column=0, padx=5, pady=5)
entry_periods = ttk.Entry(frame)
entry_periods.grid(row=2, column=1, padx=5, pady=5)

# Create a button to trigger the plot
plot_button = ttk.Button(frame, text="Plot", command=plot_graph)
plot_button.grid(row=3, column=0, columnspan=2, pady=10)

# Create a label to display errors
error_label = ttk.Label(frame, text="", foreground="red")
error_label.grid(row=4, column=0, columnspan=2, pady=5)

# Start the Tkinter main loop
window.mainloop()
