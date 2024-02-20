import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import medfilt


def grab_lidar_data(filename, location):
    """
    Function to grab lidar data from a file.

    Args:
        filename (str): The name of the file containing lidar data.
        location (str): The location of the file.

    Returns:
        list: A list of lidar data, where each element is a sublist of tuples representing (x, y) coordinates.
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


def is_point_in_quadrant(point, quadrant_bounds):
    """
    Checks if a point is within the given quadrant bounds.

    Args:
        point (tuple): A tuple containing the x and y coordinates of a point.
        quadrant_bounds (tuple): A tuple containing the x_min, x_max, y_min, and y_max of a quadrant.

    Returns:
        bool: True if the point is within the quadrant, False otherwise.
    """
    x, y = point
    x_min, x_max, y_min, y_max = quadrant_bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def trilaterate_position(quadrant, quadrants, lidar_readings_list, reading_index):
    """
    Trilaterates the position based on a specific set of LiDAR readings in the specified quadrant.

    Args:
        quadrant (int): The specified quadrant for which to triangulate the position.
        quadrants (list): A list containing the bounds for each quadrant.
        lidar_readings_list (list): A list of lists, where each inner list contains LiDAR readings as (x, y) tuples.
        reading_index (int): The index of the reading set to analyze.

    Returns:
        np.array: An array containing the trilaterated x and y coordinates.

    Raises:
        ValueError: If there are less than 3 points to triangulate a position or if the Trilateration does not converge.
    """

    # Retrieve the bounds for the specified quadrant
    quadrant_bounds = quadrants[quadrant]

    # Filter the readings from each LiDAR to include only those in the specified quadrant
    filtered_readings = []
    for readings in lidar_readings_list:
        # Check if the reading index is within the range of available readings for the current LiDAR
        if reading_index < len(readings):
            filtered_sublist = [
                point
                for point in readings[reading_index]
                if is_point_in_quadrant(point, quadrant_bounds)
            ]
            filtered_readings.extend(filtered_sublist)
        else:
            raise ValueError("Reading index out of range.")

    # Ensure there are at least three points to triangulate a position
    if len(filtered_readings) < 3:
        raise ValueError("Need at least 3 points to triangulate a position.")

    # Function to minimize the squared distances to all points
    def squared_distance_sum(coords):
        """Calculates the sum of squared distances from a point to all filtered readings."""
        x, y = coords
        return sum(
            (x - point[0]) ** 2 + (y - point[1]) ** 2 for point in filtered_readings
        )

    # Estimate initial guess for the minimize function as the mean of filtered readings
    initial_guess = np.mean(filtered_readings, axis=0)

    # Perform the minimization to find the position that minimizes squared distances to all points
    result = minimize(squared_distance_sum, initial_guess, method="Nelder-Mead")

    # Check if the optimization was successful and return the result
    if result.success:
        return result.x
    else:
        raise ValueError("Trilateration did not converge")

def plot_graph(

    x,
    y,
    squares,
    trilaterated_positions=None,
    window_size=5,
    x_min=-2,
    x_max=2,
    y_min=-2,
    y_max=2,
):
    # Apply a median filter to remove noise
    x_filtered = medfilt(x, kernel_size=window_size)
    y_filtered = medfilt(y, kernel_size=window_size)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the filtered graph as scatter plot
    ax.scatter(x_filtered, y_filtered, s=1, c="black")

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
            fontsize=10,
            color="r",
            ha="center",
            va="center",
        )

    # Plot each trilaterated position for the specified quadrants
    if trilaterated_positions is not None:
        for quadrant, position in trilaterated_positions.items():
            ax.scatter(
                position[0],
                position[1],
                s=100,  # Increased size
                c="red",  # Changed color to red
                marker="o",  # Use circle marker
                label=(
                    "Trilaterated Points"
                    if quadrant == list(trilaterated_positions.keys())[0]
                    else ""
                ),
            )
            # Display coordinates below points, larger and in red
            ax.text(
                position[0],
                position[1] - 0.05,  # Adjust to place text below the point
                f"({position[0]:.2f}, {position[1]:.2f})",
                fontsize=13,  # Larger font size
                color="red",  # Text in red
                ha="center",
                va="top",  # Adjust vertical alignment to top to place below
            )

    # Set the x and y limits of the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set the aspect ratio to equal
    ax.set_aspect("equal")

    # Set the x and y labels and title of the plot
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("LIDAR Mapping")

    # Add a legend if there are trilaterated positions
    if trilaterated_positions is not None:
        ax.legend(loc="upper right")

    # Show the plot
    plt.show()


# ----------------------------------- Usage ---------------------------------- #
print("Running the Trilateration Algorithm...")
quadrants = split_graph()  # Generate the quadrant boundaries

quadrants_indexes = [1, 3, 9, 14]  # The indexes of the quadrants to check

lidar_readings_list = [
    grab_lidar_data("corrected_lidar_1.txt", "data\\run_2"),
    grab_lidar_data("corrected_lidar_2.txt", "data\\run_2"),
    grab_lidar_data("corrected_lidar_3.txt", "data\\run_2"),
]

for reading_index in range(len(lidar_readings_list[0])):
    trilaterated_positions = {}  # Store trilaterated positions for each quadrant
    for quadrant in quadrants_indexes:
        try:
            person_position = trilaterate_position(
                quadrant, quadrants, lidar_readings_list, reading_index
            )
            print(f"Trilaterated Position for reading {reading_index}: {person_position}")
            trilaterated_positions[quadrant] = person_position
        except ValueError as e:
            print(e)

    # Extract x and y coordinates from the specified reading_index across all LiDAR readings
    x = [point[0] for readings in lidar_readings_list for point in readings[reading_index]]
    y = [point[1] for readings in lidar_readings_list for point in readings[reading_index]]

    # Plot the graph with the trilaterated positions for the current reading_index
    plot_graph(x, y, quadrants, trilaterated_positions=trilaterated_positions)