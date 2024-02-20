import math
import numpy as np
import re

# ---------------------------------------------------------------------------- #
#            Util file containing useful functions for data analysis           #
# ---------------------------------------------------------------------------- #

# ----------------------------- Parsing read data ---------------------------- #
def filter_readings(readings, x_min, x_max, y_min, y_max):
    """
    Args:
        readings (list): list of readings theta and r.
        x_min (double): minimum value
        x_max (double): maximum value
        y_min (double): minimum value
        y_max (double): maximum value

    Returns:
        list: returns a list of points with the specified range
    """    
    filtered_readings = []
    for thetas, dists in readings:
        filtered_thetas = []
        filtered_dists = []
        for theta, dist in zip(thetas, dists):
            x = dist * math.cos(theta)
            y = dist * math.sin(theta)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                filtered_thetas.append(theta)
                filtered_dists.append(dist)
        if filtered_thetas and filtered_dists:
            filtered_readings.append((filtered_thetas, filtered_dists))
    return filtered_readings

def identify_clusters(x, y, threshold=0.1, min_points=2):
    """
    The function `identify_clusters` takes in two lists of x and y coordinates, and groups points that are
    within a certain threshold distance of each other, returning a list of grouped points.
    
    :param x: The x-coordinates of the points
    :param y: The y parameter represents the y-coordinates of the points
    :param threshold: The threshold parameter determines the maximum distance allowed between two points
    for them to be considered part of the same group. If the distance between two points is less than
    the threshold, they will be grouped together
    :param min_points: The `min_points` parameter specifies the minimum number of points required for a
    group to be considered valid. If a group has fewer points than the `min_points` value, it will not
    be included in the final result, defaults to 2 (optional)
    :return: a list of grouped points. Each group is represented as a list of two lists: one containing
    the x-coordinates of the points in the group, and the other containing the y-coordinates of the
    points in the group.
    """

    grouped_points = []
    visited = set()

    for i in range(len(x)):
        if i in visited:
            continue

        current_group_x = [x[i]]
        current_group_y = [y[i]]
        visited.add(i)

        for j in range(i+1, len(x)):
            if j in visited:
                continue

            distance = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if distance < threshold:
                current_group_x.append(x[j])
                current_group_y.append(y[j])
                visited.add(j)

        if len(current_group_x) >= min_points:
            grouped_points.append([current_group_x, current_group_y])

    return grouped_points

# ----------------------- Converting coordinate systems ---------------------- #
def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates (r, theta) to Cartesian coordinates (x, y).
    
    Arguments:
    r -- a list or array of distances from the origin to the points
    theta -- a list or array of angles (in radians) between the positive x-axis and the line segments from the origin to the points
    
    Returns:
    Two arrays (x, y) representing the Cartesian coordinates of the points.
    """
    theta = np.radians(theta) # Convert degrees to radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y

# --------------------------- Reading RPLidar data --------------------------- #
def read_sample_slamtec(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    thetas = []
    dists = []
    for line in lines:
        theta = re.findall(r'theta: (\d+\.\d+)', line)
        dist = re.findall(r'Dist: (\d+\.\d+)', line)
        if theta and dist:
            thetas.append(float(theta[0]))
            dists.append(float(dist[0]) * 0.001)
    return thetas, dists

def read_multiple_samples_slamtec(filename):
   # This code reads data from a file containing RPLidar readings and parses it into a list of tuples,
   # where each tuple contains two lists: one for the angles (thetas) and one for the distances
   # (dists) of the readings. The function `re.findall()` is used to extract the theta and distance
   # values from each line of the file, and the `if theta and dist` statement checks if both values
   # were successfully extracted before appending them to the `thetas` and `dists` lists. The code
   # also checks if the theta value goes back to 0 and starts a new reading in that case. Finally, the
   # function returns a list of all the readings.
   
    with open(filename, 'r') as f:
        lines = f.readlines()
    readings = []
    thetas = []
    dists = []
    last_theta = None
    for line in lines:
        theta = re.findall(r'theta: (\d+\.\d+)', line)
        dist = re.findall(r'Dist: (\d+\.\d+)', line)
        if theta and dist:
            theta_val = float(theta[0])
            if last_theta is not None and theta_val < last_theta:
                # Start a new reading if theta goes back to 0
                readings.append((thetas, dists))
                thetas = []
                dists = []
            thetas.append(theta_val)
            dists.append(float(dist[0]) * 0.001)
            last_theta = theta_val
    if thetas and dists:
        readings.append((thetas, dists))
    if readings:
        del readings[-1]  # Delete the last reading if more than one reading
    return readings

# --------------------------------- Quadrants -------------------------------- #
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

def get_single_point_square_index(x, y, squares):
    """
    Args:
        x (float): x point
        y (float): y point
        squares (list): list of squares

    Returns:
        int: square index
    """
    # Loop through each square
    for i, square in enumerate(squares):
        # Check if the point is within the square
        if x >= square[0] and x <= square[1] and y >= square[2] and y <= square[3]:
            # Return the index of the square
            return i
    
    # If the point is not within any of the squares, return None
    return None

def get_square_index(x, y, squares):
    """
    Args:
        x (list): x points
        y (list): y points
        squares (list): list of squares

    Returns:
        int: square index
    """
    
    # Initialize a list to hold the number of points in each square
    point_counts = [0] * len(squares)

    # Loop through each point and determine which square it is in
    for i in range(len(x)):
        for j in range(len(squares)):
            if squares[j][0] <= x[i] <= squares[j][1] and squares[j][2] <= y[i] <= squares[j][3]:
                point_counts[j] += 1

    # Find the index of the square with the most points
    max_count = max(point_counts)
    max_index = point_counts.index(max_count)

    # Return the index of the square with the most points
    return max_index

def get_points_in_square(x, y, squares, square_index):
    # Get the boundaries of the specified square
    square_bounds = squares[square_index]

    # Initialize lists to hold the coordinates of the points within the square
    x_points = []
    y_points = []

    # Loop through each point and check if it falls within the square
    for i in range(len(x)):
        if square_bounds[0] <= x[i] <= square_bounds[1] and square_bounds[2] <= y[i] <= square_bounds[3]:
            x_points.append(x[i])
            y_points.append(y[i])

    # Return the lists of points within the square
    return x_points, y_points

def find_clusters_square_index(squares, clusters):
    """
    The function `find_clusters_square_index` takes in a list of squares and a list of grouped points,
    and returns a list of square indices where the majority of points in each group lie.
    
    :param squares: The `squares` parameter is a list of tuples, where each tuple represents the
    coordinates of a square. Each tuple contains four values: `x1`, `x2`, `y1`, and `y2`, which
    represent the minimum and maximum x and y coordinates of the square, respectively
    :param clusters: clusters is a list of tuples, where each tuple represents a group of
    points. Each tuple contains two lists, group_x and group_y, which represent the x-coordinates and
    y-coordinates of the points in that group, respectively
    :return: The function `find_clusters_square_index` returns a list of square indices.
    """

    square_indices_for_clusters = []
    
    for group in clusters:
        group_x, group_y = group
        square_count = {}

        for x, y in zip(group_x, group_y):
            for i, (x1, x2, y1, y2) in enumerate(squares):
                if x1 <= x < x2 and y1 <= y < y2:
                    square_count[i] = square_count.get(i, 0) + 1
                    break
        
        # Find the square index where the majority of points lie
        majority_square_index = max(square_count, key=square_count.get)
        square_indices_for_clusters.append(majority_square_index)

    return square_indices_for_clusters