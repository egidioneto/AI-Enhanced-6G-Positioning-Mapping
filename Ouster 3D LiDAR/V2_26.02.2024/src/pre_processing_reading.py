from ouster import client
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import save_load_tfrecord as slxyz

# ---------------------------------------------------------------------------- #
#             LiDAR code for extraction and preprocessing the data             #
# ---------------------------------------------------------------------------- #

# ----------------------------- Global parameters ---------------------------- #
# Sensor UDP port numbers
UDP_PORT_LIDAR = 7502
UDP_PORT_IMU = 7503
# Operating mode
OPERATING_MODE = client.OperatingMode.OPERATING_NORMAL

# --------------------------------- Functions -------------------------------- #
def configure_and_connect_sensor(hostname):
    """
    The function `configure_and_connect_sensor` configures and connects to a sensor using the provided
    hostname.
    
    Args:
      hostname: The hostname parameter is the IP address or hostname of the sensor device you want to
    connect to.
    
    Returns:
      the "source" object, which is an instance of the client.Sensor class.
    """
    
    print("Configuring and connecting to the sensor...")
    try:
        config = client.SensorConfig()
        config.udp_port_lidar = UDP_PORT_LIDAR
        config.udp_port_imu = UDP_PORT_IMU
        config.operating_mode = OPERATING_MODE

        client.set_config(hostname, config, persist=True, udp_dest_auto=True)
        source = client.Sensor(hostname, UDP_PORT_LIDAR, UDP_PORT_IMU)
        print("Successfully configured and connected.")
        return source
    except Exception as e:
        print(f"Failed to configure or connect to the sensor. Error: {e}")
        return None

def get_xyz_from_source(source):
    """
    The function `get_xyz_from_source` takes a source object, extracts metadata and scans from it,
    applies an XYZ lookup table to each scan, and returns a list of coordinates.
    
    Args:
      source: The "source" parameter is the input source from which the XYZ coordinates are obtained. It
    could be a file, a database, or any other source that contains the necessary data for generating XYZ
    coordinates.
    
    Returns:
      a list of coordinates in the form of tuples (x, y, z).
    """
    metadata = source.metadata
    scans = client.Scans(source)
    xyzlut = client.XYZLut(metadata)
    coordinates = []

    for scan in scans:
        if scan is not None:
            xyz = xyzlut(scan)
            x, y, z = [c.flatten() for c in np.dsplit(xyz, 3)]
            coordinates.append((x, y, z))

    return coordinates

def filter_xyz(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
  """
  The function filters readings based on specified minimum and maximum values for x, y, and z.
  
  Args:
    x: An array of x-coordinates of readings.
    y: The parameter "y" is a list or array containing the values of the y-coordinate for each data
  point.
    z: The parameter "z" represents an array of readings or values.
    x_min: The minimum value for the x-axis range.
    x_max: The maximum value for the x-axis that you want to include in the filtered data.
    y_min: The minimum value for the y-axis in the filter.
    y_max: The maximum value for the y-axis.
    z_min: The minimum value for the z-axis.
    z_max: The maximum value for the z-axis.
  
  Returns:
    three arrays: x[mask], y[mask], and z[mask]. These arrays contain the values from the original x,
  y, and z arrays that satisfy the given conditions.
  """
  x = np.array(x)
  y = np.array(y)
  z = np.array(z)
  mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
  return x[mask], y[mask], z[mask]

def visualize_with_simple_viz(sensor_hostname):
  """
  The function `visualize_with_simple_viz` runs the `ouster-cli` command with the specified sensor
  hostname and the "viz" argument to visualize the sensor data.
  
  Args:
    sensor_hostname: The sensor_hostname parameter is a string that represents the hostname or IP
  address of the sensor you want to visualize.
  """
  command = ["ouster-cli", "source", sensor_hostname, "viz"]
  subprocess.run(command)

def plot_3d_realtime(x, y, z, interval=1000):
  """
  The function `plot_3d_realtime` plots a 3D scatter plot in real-time using the given x, y, and z
  coordinates.
  
  Args:
    x: A list or array of x-coordinates for the points to be plotted in the 3D plot.
    y: The parameter `y` represents the values for the y-axis in the 3D plot. It is a list or array of
  values corresponding to each point in the plot.
    z: The parameter `z` represents the values for the z-axis in the 3D plot. It is a list or array of
  numerical values that correspond to the z-coordinates of the points to be plotted.
    interval: The `interval` parameter specifies the time interval (in milliseconds) between each
  frame update in the animation. By default, it is set to 1000 milliseconds (1 second). You can adjust
  this value to control the speed of the animation. Defaults to 1000
  """
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  def update(num):
      ax.clear()
      ax.scatter(x[:num], y[:num], z[:num])
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

  ani = FuncAnimation(fig, update, frames=len(x), interval=interval)
  plt.show()

def initialize_3d_plot():
  """
  The function initializes a 3D plot using matplotlib in Python.
  
  Returns:
    a figure object and an axis object.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plt.ion()  # Turn on interactive mode
  plt.show()
  return fig, ax

def update_3d_plot(ax, x, y, z):
  """
  The function `update_3d_plot` updates a 3D plot by clearing the axes, plotting new data points, and
  updating the labels.
  
  Args:
    ax: The ax parameter is the Axes3D object on which the scatter plot will be created. It represents
  the 3D plot on which the data points will be plotted.
    x: The x parameter represents the x-coordinates of the points in the 3D plot.
    y: The parameter `y` represents the values for the y-axis in the 3D plot.
    z: The parameter `z` represents the values for the z-axis in the 3D plot. It is a list or array of
  numerical values that correspond to the z-coordinates of the points to be plotted.
  """
  ax.clear()
  ax.scatter(x, y, z)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.draw()
  plt.pause(0.1)

# Helper function to generate filename
from datetime import datetime
def generate_filename():
    now = datetime.now()
    return f"xyz_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"

# ----------------------------------- Usage ---------------------------------- #
# Grabbing sensor source and metadata
source = configure_and_connect_sensor('os-122308001777') # hostname of your sensor

try:
  fig, ax = initialize_3d_plot()
  
  while True:
    try:
      # ------------------- Grabbing x,y,z arrays from the source ------------------ #
      x, y, z = get_xyz_from_source(source)

      # Define your x, y, z coordinate limits for filtering
      x_min, x_max = -4, 4
      y_min, y_max = -4, 4
      z_min, z_max = -4, 4

      # Filter the coordinates
      x, y, z = filter_xyz(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max)
      
      # ------------------------- Saving the data to a file ------------------------ #
      slxyz.save_xyz_to_tfrecord(filename=generate_filename(), 
                             directory='ouster\\data\\xyz-data\\tfrecord', x=x, y=y, z=z)
      
      # --------------------------- Plotting in real-time -------------------------- #
      # Update the plot in real-time
      update_3d_plot(ax, x, y, z)
        
    except Exception as e:
      print(f"An error occurred: {e}")
except KeyboardInterrupt:
  print("Operation stopped by the user.")