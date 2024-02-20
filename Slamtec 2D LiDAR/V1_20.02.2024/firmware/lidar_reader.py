"""
About this code:

The code reads data from a Lidar sensor via a named pipe, and plots it in 
real-time using Matplotlib. The Lidar data is processed to convert polar 
coordinates to Cartesian coordinates, and a median filter is applied to 
remove noise before plotting. The program runs in two separate threads, 
one for collecting data and the other for plotting.
"""

# ------------------------ Import necessary libraries ------------------------ #
import os
import subprocess
import numpy as np
import threading
import time

import matplotlib.pyplot as plt
from scipy.signal import medfilt

# ---------------- Create an empty list to hold the Lidar data --------------- #
readings = []

# ---------- Call the C++ program to read data from the Lidar sensor --------- #
def runCpp():
    cmd = ['sudo', './ultra_simple', '--channel', '--serial', '/dev/ttyUSB0', '115200']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read output and error streams
    out, err = p.communicate()

    # Print output and error
    print(out.decode('utf-8'))
    print(err.decode('utf-8'))

# ----------- Define a function to plot the Lidar data in real-time ---------- #
def plotData(window_size):
    # Two arrays (x, y) representing the Cartesian coordinates of the points.
    x = []
    y = []

    # to run GUI event loop
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots()
    scatter = ax.scatter(x, y, s=4, c='black')

    # setting title
    plt.title("Lidar Mapping", fontsize=15)

    # setting x-axis label and y-axis label
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    
    while True: 
        if(len(readings) != 0):
            # Get the latest data from the Lidar sensor
            theta, r = readings[-1]
            
            # Convert polar coordinates to Cartesian coordinates
            x, y = polar_to_cartesian(r, theta)
            
            # ---------------------------- Plot the Lidar data --------------------------- #
            # Apply a median filter to remove noise
            x_filtered = medfilt(x, kernel_size=window_size)
            y_filtered = medfilt(y, kernel_size=window_size)
            
            if x_filtered.any() and y_filtered.any():
                # Setting axis limits
                ax.set_xlim(min(x_filtered), max(x_filtered)) # type: ignore
                ax.set_ylim(min(y_filtered), max(y_filtered)) # type: ignore
                
                # Updating data values
                scatter.set_offsets(np.c_[x_filtered, y_filtered])

                # Drawing updated values
                figure.canvas.draw()

                # Run the GUI event loop until all UI events currently waiting have been processed
                figure.canvas.flush_events()

            # ------------------------------------- - ------------------------------------ #
        # Sleep before plotting again
        time.sleep(1) 

# ---------- Define a function to collect data from the Lidar sensor --------- #
def collectData():
    # Open the named pipe for reading
    fifo_file = '/tmp/lidar_pipe'
    if not os.path.exists(fifo_file):
        os.mkfifo(fifo_file) # type: ignore
    pipe = os.open(fifo_file, os.O_RDONLY)
    print('Reading...')
    
    # Read and parse the Lidar data from the named pipe
    last_theta = None
    thetas = []
    dists = []
    while True:     
        lidar_data_str = os.read(pipe, 1024) 
        if len(lidar_data_str) == 0:
            break
        
        # Parse the string data to extract the theta, dist, and Q values
        data = lidar_data_str.decode().split()
        try:
            theta = float(data[0])
            dist = float(data[1])* 0.001 # Distance: mm --> m
            # q = float(data[2])

            # Check if theta goes back to 0, which indicates a new reading
            if last_theta is not None and theta < last_theta:
                readings.append((thetas, dists))  # Finished one reading
                print('Reading finished...')
            
                # Reset the figure
                thetas = []
                dists = []
            else:
                thetas.append(theta)
                dists.append(dist)
            last_theta = theta
        except:
            print("Error collecting data")

    # Close the named pipe
    os.close(pipe)

# -------- Define a function to convert polar to cartesian coordinates ------- #
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
    
    if len(r) != len(theta):
        # determine which is larger and find the ratio to resize the smaller one
        larger = np.argmax([len(r), len(theta)])
        ratio = np.ceil(np.abs(len(r) - len(theta)) / np.min([len(r), len(theta)])) # type: ignore
        if larger == 0:
            # resize theta to match r
            theta = np.resize(theta, r.shape) * ratio
        else:
            # resize r to match theta
            r = np.resize(r, theta.shape) * ratio
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# ------------------------------------- - ------------------------------------ #
if __name__ == "__main__":
    #runCpp()
    
    # Collecting data
    t1 = threading.Thread(target=collectData)
    t1.start()
    
    # Plotting data:
    plotData(9);
    
    t1.join()