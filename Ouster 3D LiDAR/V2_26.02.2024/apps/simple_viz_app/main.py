import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import subprocess
import threading
import sys

def visualize_with_simple_viz(sensor_hostname=None, pcap_path=None):
    try:
        if sensor_hostname:
            source = sensor_hostname
        elif pcap_path:
            source = pcap_path
        else:
            print("Provide either sensor hostname or pcap path.")
            return

        command = ["ouster-cli", "source", source, "viz"]
        # Windows-specific settings to prevent the command prompt from appearing
        if sys.platform.startswith('win'):
            # CREATE_NO_WINDOW option prevents the command window from popping up
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess.run(command, check=True, startupinfo=startupinfo)
        else:
            subprocess.run(command, check=True)
    except Exception as e:
        return " Source type expected to be a sensor hostname or ip address,\nplease check that you can ping the sensor hostname/ip address."
    return ""  # Return an empty string to indicate success without errors


def run_visualization(hostname):
    error_message = visualize_with_simple_viz(sensor_hostname=hostname)
    if error_message:
        # If an error occurred, schedule an update to display the error message
        root.after(100, lambda: update_status(error_message, "red"))
    else:
        # Clear the status message on success
        root.after(100, lambda: update_status("", "green"))


def on_run_clicked():
    hostname = hostname_var.get()
    update_status("Loading...", "light green")
    # Run the visualization in a separate thread to prevent GUI freezing
    threading.Thread(target=run_visualization, args=(hostname,)).start()


def update_status(message, color):
    status_label.config(text=message, foreground=color)


# Initialize the main window with a theme
root = ThemedTk(theme="equilux")
root.title("Sensor Configuration")
root.geometry("800x440")  # Width x Height

icon = tk.PhotoImage(file='icon.png')
root.iconphoto(True, icon)

# Set the theme color
theme_color = "#26262b"

# Configure the root window's background color
root.configure(bg=theme_color)

# Load and place the logo image
logo_image = tk.PhotoImage(
    file="logo.png"
)  # Ensure 'logo.png' is in the correct directory
logo_placeholder = ttk.Label(root, image=logo_image, background=theme_color)
logo_placeholder.pack(pady=20)

# Customize styles for darker widgets
style = ttk.Style()
style.configure(
    "Darker.TEntry", foreground="white", background="#1c1c1e", font=("Arial", 18)
)
style.configure(
    "Darker.TButton",
    font=("Arial", 16, "bold"),
    background="#1c1c1e",
    foreground="white",
)

# Increase padding for the button to make it appear larger
style.configure("Darker.TButton", padding=(20, 10))

# Title of the application (increased text size)
title = ttk.Label(
    root,
    text="Configure Your Sensor",
    background=theme_color,
    foreground="white",
    font=("Arial", 22, "bold"),
)
title.pack()

# Instruction text (increased text size)
instruction = ttk.Label(
    root,
    text="Enter the LiDAR Hostname:",
    background=theme_color,
    foreground="white",
    font=("Arial", 12),
)
instruction.pack(pady=5)

# Input field with a pre-filled placeholder (increased font size for larger input)
hostname_var = tk.StringVar(value="os-")
hostname_entry = ttk.Entry(
    root, textvariable=hostname_var, font=("Arial", 18), width=30, style="Darker.TEntry"
)
hostname_entry.pack(pady=10)

# Button to run the code (adjusted style for larger appearance)
run_button = ttk.Button(
    root, text="Run", command=on_run_clicked, style="Darker.TButton", width=20
)
run_button.pack(pady=20)

# Status Label below the Run button for displaying status messages
status_label = ttk.Label(
    root, text="", background=theme_color, foreground="light green", font=("Arial", 12)
)
status_label.pack(pady=10)

# Start the GUI event loop
root.mainloop()
