import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_graph():
    try:
        A = float(entry_A.get())
        omega = float(entry_omega.get())
        periods = int(entry_periods.get())
        t = np.linspace(0, 2 * np.pi * periods, 1000)
        y = A * np.sin(omega * t)

        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'A sin(ωt) with A={A}, ω={omega}, and {periods} periods')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except ValueError:
        error_label.config(text="Please enter valid numeric values.")


# Create the main window
window = tk.Tk()
window.title("Sinusoidal Plotter")

# Create input fields
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

# Create plot button
plot_button = ttk.Button(frame, text="Plot", command=plot_graph)
plot_button.grid(row=3, column=0, columnspan=2, pady=10)

# Create error label
error_label = ttk.Label(frame, text="", foreground="red")
error_label.grid(row=4, column=0, columnspan=2, pady=5)

# Start the main loop
window.mainloop()
