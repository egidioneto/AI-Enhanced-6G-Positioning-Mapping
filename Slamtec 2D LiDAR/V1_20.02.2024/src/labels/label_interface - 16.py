import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime


# ----------------------------- Global Variables ----------------------------- #
current_plot = 0
selected_quadrants = []
prev_selection = [0] * 16
rectangles = []  # Add this line to declare rectangles as a global variable


# ------------------------------------- - ------------------------------------ #
def generate_random_coordinates(num_points=100, x_min=-2, x_max=2, y_min=-2, y_max=2):
    """
    Generates a list of random x, y coordinates within the given boundaries.

    Args:
        num_points (int): Number of random points to generate.
        x_min, x_max, y_min, y_max (float, optional): Boundaries for the random coordinates.

    Returns:
        list, list: Lists of x and y coordinates.
    """
    import random

    x_coords = [random.uniform(x_min, x_max) for _ in range(num_points)]
    y_coords = [random.uniform(y_min, y_max) for _ in range(num_points)]
    return x_coords, y_coords


# ----------------------------- Grabbing Readings ---------------------------- #
def grab_lidar_data(filename, location):
    filepath = os.path.join(location, filename)
    data = []
    with open(filepath, "r") as file:
        sublist = []
        for line in file:
            if line.strip() == "":
                if sublist:
                    data.append(sublist)
                    sublist = []
            else:
                parts = line.split()
                sublist.append((float(parts[0]), float(parts[1])))
        if sublist:  # add the last sublist if file does not end with a newline
            data.append(sublist)
    return data


# ------------------------------ Data Functions ------------------------------ #
def split_graph(n=16, x_min=-2, x_max=2, y_min=-2, y_max=2):
    """
    This function splits a graph into N equal squares using the provided boundaries.
    It returns a list of tuples, containing the coordinates of the four corners of each square.

    Args:
        n (int, optional):  Defaults to 16.
        x_min (int, optional): Defaults to -2.
        x_max (int, optional):  Defaults to 2.
        y_min (int, optional):  Defaults to -2.
        y_max (int, optional):  Defaults to 2.

    Returns:
        list: list of squares coordinates
    """

    width = (x_max - x_min) / np.sqrt(n)
    height = (y_max - y_min) / np.sqrt(n)

    # Initialize a list to hold the square coordinates
    squares = []

    # Loop through each row and column of squares
    for i in range(int(np.sqrt(n))):
        for j in range(int(np.sqrt(n))):
            # Calculate the x and y coordinates of the square's corners
            x1 = x_min + j * width
            x2 = x_min + (j + 1) * width
            y1 = y_min + i * height
            y2 = y_min + (i + 1) * height

            # Append the square coordinates to the list
            squares.append((x1, x2, y1, y2))

    # Return the list of square coordinates
    return squares


# Function to update data
def update_data():
    global data_queue
    if data_queue:
        x_data, y_data = get_next_data()
        plot_graph(canvas, x_data, y_data, squares)
    else:
        # If the queue is empty, print a message or close the GUI
        print("All data processed!")
        root.quit()
        root.destroy()


# ------------------------------- GUI Functions ------------------------------ #
def plot_graph(canvas, x, y, squares):
    """
    The function `plot_graph` plots a scatter plot of points and overlays rectangles on the plot,
    allowing the user to click on the rectangles.

    Args:
      canvas: The `canvas` parameter is a tkinter canvas widget where the graph will be displayed.
      x: The x parameter is a list of x-coordinates for the points to be plotted on the graph.
      y: The parameter `y` represents the y-coordinates of the points to be plotted on the graph.
      squares: The `squares` parameter is a list of tuples, where each tuple represents a square. Each
    tuple contains four values: the x-coordinate of the left side of the square, the x-coordinate of the
    right side of the square, the y-coordinate of the bottom side of the square, and the
    """
    global rectangles  # Declare rectangles as global within this function
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, c="black")
    rectangles = []  # Clear the previous rectangles
    for i, square in enumerate(squares):
        rect = plt.Rectangle(
            (square[0], square[2]),
            square[1] - square[0],
            square[3] - square[2],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        rectangles.append(rect)
        ax.text(
            (square[0] + square[1]) / 2,
            (square[2] + square[3]) / 2,
            str(i),
            fontsize=10,
            color="r",
            ha="center",
            va="center",
        )
    fig.canvas.mpl_connect("button_press_event", on_rectangle_click)
    canvas.figure = fig
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.draw()


# Function to handle rectangle click events
def on_rectangle_click(event):
    """
    The function `on_rectangle_click` checks if a mouse click event occurred within a rectangle and
    toggles the state of a checkbox associated with that rectangle.

    Args:
      event: The event parameter is an object that represents the event that occurred. In this case, it
    is likely a mouse click event on a rectangle.
    """
    global rectangles  # Declare rectangles as global within this function
    for i, rect in enumerate(rectangles):
        if rect.contains(event)[0]:
            checkbox_vars[i].set(1 - checkbox_vars[i].get())  # Toggle between 0 and 1


# Function to unselect all checkboxes
def unselect_all():
    for var in checkbox_vars:
        var.set(0)


# Function to handle the "Next" button click
def next_plot():
    global current_plot, selected_quadrants, prev_selection
    quadrant_states = [var.get() for var in checkbox_vars]
    selected_quadrants.append(quadrant_states)
    prev_selection = quadrant_states
    prev_label.config(text=f"Previous: {quadrant_states}")
    current_plot += 1
    unselect_all()  # Unselect all checkboxes
    update_data()  # This will now pull from the queue and close the GUI when done


# Function to handle the "Fill" button click
def fill_with_prev_selection():
    """
    This function sets the result for each position left in data_queue to the current selection,
    closes the GUI, and then proceeds to the subsequent code.
    """
    global data_queue, selected_quadrants

    # Capture the current selection
    current_selection = [var.get() for var in checkbox_vars]

    # Populate the remaining entries of data_queue to selected_quadrants
    while data_queue:
        selected_quadrants.append(current_selection)
        data_queue.pop(0)

    # Close the GUI
    root.quit()
    root.destroy()


# Function to handle the "Keep" button click
def keep_selection():
    """
    The function "keep_selection" restores the previous selection of checkboxes.
    """
    for i, var in enumerate(checkbox_vars):
        var.set(prev_selection[i])
    next_plot()  # Call the next_plot function to move to the next plot

def apply_selection_to_next_19():
    global data_queue, selected_quadrants, prev_selection

    # Capture the current selection
    current_selection = [var.get() for var in checkbox_vars]

    # Apply the current selection to the next 19 datasets in the queue
    for _ in range(min(19, len(data_queue))):
        selected_quadrants.append(current_selection)
        data_queue.pop(0)

    # Update the previous selection
    prev_selection = current_selection

    # Update the GUI to show the next dataset if available
    if data_queue:
        next_plot()
    else:
        print("No more data in the queue.")
        root.quit()
        root.destroy()


# Function to get the next data set from the queue
def get_next_data():
    global data_queue
    if data_queue:
        data_list = data_queue.pop(0)
        x_data = [point[0] for point in data_list]  # Extract x-coordinates
        y_data = [point[1] for point in data_list]  # Extract y-coordinates
        return x_data, y_data
    else:
        return None, None  # Return None if the queue is empty


# ---------------------------------- Results --------------------------------- #
def save_results_to_txt(selected_quadrants, filename="results"):
    """
    The function `save_results_to_txt` saves a list of selected quadrants to a text file, with each
    quadrant represented as a comma-separated string on a new line. The current time is appended to
    the filename to ensure uniqueness.

    Args:
      selected_quadrants: selected_quadrants is a list of quadrants that you want to save to a text
    file. Each quadrant is represented as a list of values.
      filename: The `filename` parameter is a string that specifies the base name of the file where
    the results will be saved. The current time is appended to this name. Defaults to "results".
    """
    # Get the current time and format it as a string
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{current_time}.txt"

    with open(full_filename, "w") as f:
        for quadrants in selected_quadrants:
            line = ",".join(map(str, quadrants))
            f.write(f"{line}\n")


# ----------------------------- Grabbing the data ---------------------------- #
# Define your queue of data. For the sake of the example, I'm generating 5 random sets of coordinates
# data_queue = [generate_random_coordinates() for _ in range(5)]
data_queue = grab_lidar_data('corrected_all_4_lidars.txt', 'data\\run_2')

#! data_queue will be list of lists of tuples, where each tuple is a coordinate pair, so there is going to be a while loop here, after the loop we take the results and save it to a file

#! ------------------------ First data initialization ----------------------- !#
# Here we initialize the squares and data for the first plot
squares = split_graph(n=16)

# Pull the first set of data from the queue for the initial plot
x_data, y_data = get_next_data()

# ---------------------- Initialize GUI and run mainloop --------------------- #
# Initialize Tkinter root window
root = tk.Tk()
root.title("Quadrant Selector")

# Create canvas for plot
canvas_frame = ttk.Frame(root)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas = FigureCanvasTkAgg(plt.Figure(), canvas_frame)
plot_graph(canvas, x_data, y_data, squares)

# Create checkboxes for quadrants
checkbox_vars = [tk.IntVar() for _ in range(16)]
checkbox_frame = ttk.Frame(root)
checkbox_frame.pack(side=tk.TOP, fill=tk.X)
for i in range(16):
    chk = ttk.Checkbutton(checkbox_frame, text=f"Q {i}", variable=checkbox_vars[i])
    chk.grid(row=i // 4, column=i % 4, sticky=tk.W + tk.E)

# Create Next, Keep and Fill buttons
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)
next_button = ttk.Button(button_frame, text="Next", command=next_plot)
next_button.pack(side=tk.LEFT, padx=5, pady=5)
keep_button = ttk.Button(button_frame, text="Keep", command=keep_selection)
keep_button.pack(side=tk.LEFT, padx=5, pady=5)
fill_button = ttk.Button(
    button_frame, text="Fill All", command=fill_with_prev_selection
)
fill_button.pack(side=tk.LEFT, padx=5, pady=5)
mark_button = ttk.Button(button_frame, text="Mark Next 20", command=apply_selection_to_next_19)
mark_button.pack(side=tk.LEFT, padx=5, pady=5)


# Create label for previous selection
prev_label = ttk.Label(button_frame, text="Previous: None")
prev_label.pack(side=tk.RIGHT, padx=5, pady=5)

root.mainloop()

# --------------------------- Grabbing the results --------------------------- #
# Print selected quadrants for each plot
# The 'selected_quadrants' is a list of lists, where each list represents the
# selected quadrants for a plot. That is, this is the result of the user's input.
print("Selected quadrants length:", len(selected_quadrants))
print("Selected quadrants for each plot:", selected_quadrants)
result = selected_quadrants
save_results_to_txt(result)

# from utils import save_load_tfrecord as sltf

# sltf.save_labels_to_tfrecord(
#     filename="test_quadrants", directory=".", list_to_be_saved=selected_quadrants
# )

# loaded_selected_quadrants = sltf.load_labels_from_tfrecord(
#     filename="./test_quadrants.tfrecord"
# )
# print("Loaded Selected Quadrants:", loaded_selected_quadrants)
