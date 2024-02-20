import os
from utils import CoordinatesHandler as ch
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
from matplotlib.widgets import Button


def plot_2d_graph_with_squares(x, y, squares, ax):
    # Clear the current axes
    ax.clear()

    # Plot the data with black points of size 1
    ax.scatter(x, y, color="black", s=1)

    # Add each square to the plot as a rectangle with square index
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
        ax.text(
            (square[0] + square[1]) / 2,
            (square[2] + square[3]) / 2,
            str(i),
            fontsize=6,
            color="r",
            ha="center",
            va="center",
        )

    # Set labels, title, and grid
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("LIDAR Mapping")
    ax.grid(True)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-8, 5)

    # Set the aspect ratio to equal
    ax.set_aspect("equal")

    # Redraw the canvas to reflect updates
    plt.draw()


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


def filter_readings_xy(readings, x_min, x_max, y_min, y_max):
    """
    Filter readings based on x and y coordinates for a list of lists of tuples.

    Args:
        readings (list of lists of tuples): List of scans, where each scan is a list of (x, y) points.
        x_min (float): Minimum x value.
        x_max (float): Maximum x value.
        y_min (float): Minimum y value.
        y_max (float): Maximum y value.

    Returns:
        list of lists of tuples: List of scans with points that fall within the specified range.
    """
    filtered_scans = []
    for scan in readings:
        filtered_scan = [
            point
            for point in scan
            if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
        ]
        filtered_scans.append(filtered_scan)
    return filtered_scans


# ----------------------------- Grabbing Readings ---------------------------- #
def list_txt_files_in_folder(folder_path):
    """
    The function `list_txt_files_in_folder` lists .txt files in a folder, sorted by the time encoded in the filename.

    Args:
      folder_path: A string representing the path to the folder.

    Returns:
      A list of file paths for all the .txt files in the specified folder, sorted by the encoded time.
    """
    txt_files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".txt")
    ]

    # Sort the list by the timestamp in the filename
    # Extracting the last part of the path and then taking the timestamp part
    txt_files.sort(
        key=lambda filename: int(os.path.basename(filename).split("_")[2].split(".")[0])
    )
    return txt_files


def transform_point(point, translation):
    # Apply translation to the point
    return (point[0] + translation[0], point[1] + translation[1])


def merge_lidar_readings(readings, translations):
    """
    Merge LiDAR readings from different vertices of a square.

    :param readings: A list of lists of lists of tuples, where each sublist represents a LiDAR's readings.
    :param translations: A list of tuples representing the translation needed for each LiDAR.
    :return: A list of lists of tuples representing the merged points in the global coordinate system.
    """
    merged_scans = []

    for scan_set in zip(*readings):
        merged_scan = []
        for single_scan, translation in zip(scan_set, translations):
            for point in single_scan:
                transformed_point = transform_point(point, translation)
                merged_scan.append(transformed_point)
        merged_scans.append(merged_scan)
    create_plotting_interface(merged_scans, save_image=True)  #! Remove this line

    return merged_scans


def read_all_readings(lidar_files, correction):
    readings = []
    for filename in lidar_files:
        polar_readings = ch.read_multiple_samples_slamtec(filename)
        for thetas, dists in polar_readings:
            # Filter readings based on the theta values
            filtered_dists = []
            filtered_thetas = []
            for theta, dist in zip(thetas, dists):
                # if you want to filter based on theta:
                if 0 <= theta <= 360:
                    filtered_thetas.append(theta + correction)
                    filtered_dists.append(dist)
            x, y = ch.polar_to_cartesian(filtered_dists, filtered_thetas)
            x = x * -1.0  # Correct the direction
            readings.append(list(zip(x, y)))
    return readings


# ------------------------------ Interface part ------------------------------ #
image_counter = 0


def create_plotting_interface(
    list_of_lists_of_tuples, save_image=False, image_size=(10, 10), dpi=300, n=64
):
    # Initialize a figure for plotting
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, left=0.30, right=0.60)

    # This function will be called when the button is clicked
    def plot_next(event):
        global image_counter
        if plot_next.counter < len(list_of_lists_of_tuples):
            # Clear the current axes and plot the next set of points
            ax.clear()
            # This ensures that only items which are tuples or lists are unpacked
            x, y = zip(
                *[
                    (item[0], item[1])
                    for item in list_of_lists_of_tuples[plot_next.counter]
                    if isinstance(item, (tuple, list)) and len(item) > 1
                ]
            )
            squares = split_graph(
                n=n
            )  # Use your default parameters or adjust as needed
            plot_2d_graph_with_squares(x, y, squares, ax=ax)

            # Save the plot as an image if save_image is True
            if save_image:
                # Temporarily remove the button before saving the image
                button_ax.set_visible(False)
                plt.savefig(f"plot_{image_counter}.png", bbox_inches="tight", dpi=dpi)
                # Make the button visible again
                button_ax.set_visible(True)
                image_counter += 1

            plot_next.counter += 1
        else:
            print("No more sets to plot.")

    # Counter for the plot_next function to track the current index
    plot_next.counter = 0

    # Create a button and assign the plot_next function to be called on click
    button_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    button = Button(button_ax, "Next")
    button.on_clicked(plot_next)

    # Show the initial plot or the first set of points
    plot_next(None)

    # Show the plot with the button
    plt.show()


# ----------------------- Saving and reading the result ---------------------- #
def save_to_file(data, filename, location):
    """
    The function `save_to_file` saves data to a file in a specified location, with each sublist of data
    written on a new line and each item within the sublist separated by a space.

    Args:
      data: The `data` parameter is a list of lists. Each sublist represents a group of items that you
    want to save to the file. Each item in the sublist is a tuple containing two values: the first value
    is the item's name and the second value is the item's quantity.
      filename: The filename parameter is the name of the file you want to save the data to. It should
    be a string value.
      location: The `location` parameter is the directory where you want to save the file. It should be
    a string representing the path to the directory.
    """
    filepath = os.path.join(location, filename)
    with open(filepath, "w") as file:
        for sublist in data:
            for item in sublist:
                file.write(f"{item[0]} {item[1]}\n")
            file.write("\n")


def read_from_file(filename, location):
    """
    The function `read_from_file` reads data from a file and returns it as a list of sublists, where
    each sublist contains pairs of floating-point numbers.

    Args:
      filename: The filename parameter is the name of the file you want to read from. It should include
    the file extension (e.g., "data.txt").
      location: The `location` parameter is the directory where the file is located. It should be a
    string representing the path to the directory.

    Returns:
      a list of sublists, where each sublist contains tuples of two floating-point numbers.
    """
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


# --------------------------------- Main code -------------------------------- #
# Define translations for each LiDAR based on its vertex position
translations = [
    (2, 2),  # Translation for LiDAR at vertex (2, 2) - LiDAR 1
    (-2, 2),  # Translation for LiDAR at vertex (-2, 2) - LiDAR 2
    (-2, -2),  # Translation for LiDAR at vertex (-2, -2) - LiDAR 3
    (2, -2),  # Translation for LiDAR at vertex (2, -2) - LiDAR 4
]

print("Reading LiDAR 1 readings...")
readings_lidar_1 = read_all_readings(
    list_txt_files_in_folder("slamtec\\samples\\run_2\\lidar_1"), correction=311
)

print("Reading LiDAR 2 readings...")
readings_lidar_2 = read_all_readings(
    list_txt_files_in_folder("slamtec\\samples\\run_2\\lidar_2"), correction=231
)

print("Reading LiDAR 3 readings...")
readings_lidar_3 = read_all_readings(
    list_txt_files_in_folder("slamtec\\samples\\run_2\\lidar_3"), correction=122
)

print("Reading LiDAR 4 readings...")
readings_lidar_4 = read_all_readings(
    list_txt_files_in_folder("slamtec\\samples\\run_2\\lidar_4"), correction=50
)

print("Merging all 4 LiDAR readings...")
all_readings = [readings_lidar_1, readings_lidar_2, readings_lidar_3, readings_lidar_4]
merged_readings = merge_lidar_readings(all_readings, translations)
# filtered_merged_readings = filter_readings_xy(merged_readings, -2, 2, -2, 2)
# create_plotting_interface(filtered_merged_readings, save_image=True)

print("Only LiDAR 1 readings...")
all_readings = [readings_lidar_1]
merged_readings = merge_lidar_readings(all_readings, [(2, 2)])
# filtered_merged_readings = filter_readings_xy(merged_readings, -2, 2, -2, 2)
# create_plotting_interface(filtered_merged_readings, save_image=True)

print("Only LiDAR 2 readings...")
all_readings = [readings_lidar_2]
merged_readings = merge_lidar_readings(all_readings, [(-2, 2)])
# filtered_merged_readings = filter_readings_xy(merged_readings, -2, 2, -2, 2)
# create_plotting_interface(filtered_merged_readings, save_image=True)

print("Only LiDAR 3 readings...")
all_readings = [readings_lidar_3]
merged_readings = merge_lidar_readings(all_readings, [(-2, -2)])
# filtered_merged_readings = filter_readings_xy(merged_readings, -2, 2, -2, 2)
# create_plotting_interface(filtered_merged_readings, save_image=True)

print("Only LiDAR 4 readings...")
all_readings = [readings_lidar_4]
merged_readings = merge_lidar_readings(all_readings, [(2, -2)])
# filtered_merged_readings = filter_readings_xy(merged_readings, -2, 2, -2, 2)
# create_plotting_interface(filtered_merged_readings, save_image=True)

# Save to file
# save_to_file(filtered_merged_readings, 'filtered_merged_readings.txt', 'slamtec\\samples\\run_2')

# Load from file
# readings_back = read_from_file('filtered_merged_readings.txt', 'slamtec\\samples\\run_2')

# print(f'Loaded file length: {len(readings_back)}')


"""
Correction:

All -x

Lidar 1: +315 
Lidar 2: +225
Lidar 3: +135
Lidar 4: +45
"""
