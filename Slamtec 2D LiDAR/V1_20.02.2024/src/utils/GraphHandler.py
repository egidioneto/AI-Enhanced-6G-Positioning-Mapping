import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from utils import CoordinatesHandler as ch

# ---------------------------------------------------------------------------- #
#                 Util file containing functions to plot graphs                #
# ---------------------------------------------------------------------------- #

def plot_difference(x_original, y_original, x_new, y_new, difference):
    """Function to plot the difference between the original and the new data"""
    plt.plot(x_original, y_original)
    plt.plot(x_new, y_new)
    plt.scatter(x_new, y_new, data=difference) # Plotting different points
    plt.show()


# --------------------------------- 3d plots --------------------------------- #
def plot_3d(x, y, z):
    """Plotting points 3d"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o') # type: ignore
    plt.show()

def plot_3d_intensity(x, y, intensity):
    """Plotting points 3d, using intensity"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, intensity, c='r', marker='o') # type: ignore
    plt.show()

# --------------------------------- 2d plots --------------------------------- #
def plot_2d_XY_continuous(x, y):
    """Plotting points 2d"""
    plt.plot(x, y)
    plt.show()
    
def plot_2d_XY_scatter(x, y, limDisplacement=0):
    """
    Plots pairs of points from two arrays of x and y coordinates using the scatter function.
    
    Arguments:
    x -- an array of x coordinates
    y -- an array of y coordinates
    """
    
    # create the plot
    fig, ax = plt.subplots()
    ax.scatter(x[0::2], y[0::2], c='blue', marker='o')  # type: ignore # plot the even-indexed pairs of points
    ax.scatter(x[1::2], y[1::2], c='blue', marker='o') # type: ignore # plot the odd-indexed pairs of points
    
    # set the axis limits and labels
    ax.set_xlim(min(x)-limDisplacement, max(x)+limDisplacement)
    ax.set_ylim(min(y)-limDisplacement, max(y)+limDisplacement)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # show the plot
    plt.show()
    
def plot_2d_XY_filtered(r, theta, window_size=5):
    # Convert polar coordinates to Cartesian coordinates
    theta = np.radians(theta) # Convert degrees to radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Apply a median filter to remove noise
    x_filtered = medfilt(x, kernel_size=window_size)
    y_filtered = medfilt(y, kernel_size=window_size)

    # Create a 2D plot using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(x_filtered, y_filtered, s=1, c='black')

    # Customize the plot
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('LIDAR Mapping')

    # Display the plot
    plt.show()

def plot_2d_graph_with_squares(x, y, squares, window_size=5, x_min=-2, x_max=2, y_min=-2, y_max=2):
    """
    Args:
        x (_type_): _description_
        y (_type_): _description_
        squares (list): a list of tuples representing the squares as (x_min, x_max, y_min, y_max)
        window_size (int, optional): _description_. Defaults to 5.
    """
    # Apply a median filter to remove noise
    x_filtered = medfilt(x, kernel_size=window_size)
    y_filtered = medfilt(y, kernel_size=window_size)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the filtered graph as scatter plot
    ax.scatter(x_filtered, y_filtered, s=1, c='black')

    # Add each square to the plot as a rectangle with square index
    for i, square in enumerate(squares):
        rect = plt.Rectangle((square[0], square[2]), square[1]-square[0], square[3]-square[2], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text((square[0]+square[1])/2, (square[2]+square[3])/2, str(i), fontsize=10, color='r', ha='center', va='center')

    # Plot the origin point as a bigger green dot
    ax.scatter(0, 0, s=20, c='black')

    # Set the x and y limits of the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set the aspect ratio to equal
    ax.set_aspect('equal')

    # Set the x and y labels and title of the plot
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('LIDAR Mapping')

    # Show the plot
    plt.show()

# -------------------------------- Polar plots ------------------------------- #
def plot_2d_polar_filtered(r, theta, window_size=5):
    """
    Plot a polar mapping, but with reduced noise
    """
    theta = np.radians(theta) # Convert degrees to radians
    
    # Apply a median filter to remove noise
    r_filtered = medfilt(r, kernel_size=window_size)

    # Create a polar plot using Matplotlib
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r_filtered, '.', markersize=1, color='black')

    # Customize the plot
    ax.set_rmax(np.max(r_filtered)) # type: ignore
    ax.set_rlabel_position(-22.5) # type: ignore
    ax.set_title('LIDAR Mapping')
    ax.grid(True)

    # Display the plot
    plt.show()

def plot_2d_polar(r, theta):
    """
    Plots points from two arrays of polar coordinates using the scatter function in a polar plot.
    
    Arguments:
    r -- an array of distances from the origin
    theta -- an array of angles (in radians) between the positive x-axis and the line segments from the origin to the points
    """
    theta = np.radians(theta) # Convert degrees to radians
    
    # create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # plot the points
    ax.scatter(theta, r, c='red', marker='o') # type: ignore
    
    # set the axis limits and labels
    ax.set_rlim(0, max(r)) # type: ignore
    ax.set_theta_zero_location('N') # type: ignore
    ax.set_theta_direction(-1) # type: ignore
    
    # show the plot
    plt.show()

# --------------------------------- Real time -------------------------------- #
def plot_XY_real_time(x, y):
    """
    Plots pairs of points from two arrays of x and y coordinates using the scatter function.
    
    Arguments:
    x -- an array of x coordinates
    y -- an array of y coordinates
    """
    
    # to run GUI event loop
    plt.ion()
    
    # create the plot
    fig, ax = plt.subplots()
    ax.scatter(x[0::2], y[0::2], c='blue', marker='o')  # type: ignore # plot the even-indexed pairs of points
    ax.scatter(x[1::2], y[1::2], c='blue', marker='o') # type: ignore # plot the odd-indexed pairs of points


    # set the axis limits and labels
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # drawing updated values
    fig.canvas.draw()
 
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()