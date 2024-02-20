from ouster import client
import socket
import time
import numpy as np
import json

# ---------------------------------------------------------------------------- #
#               This code is the client, that is, the data sender              #
# ---------------------------------------------------------------------------- #

def configure_and_connect_sensor(hostname):
    """
    Configures the sensor based on global parameters and establishes a connection.

    Args:
      hostname (str): The hostname or IP address of the Ouster lidar sensor.

    Returns:
      client.Sensor: The connected sensor source.
      None: If the connection or configuration fails.
    """
    # UDP port numbers
    UDP_PORT_LIDAR = 7502
    UDP_PORT_IMU = 7503
    # Operating mode
    OPERATING_MODE = client.OperatingMode.OPERATING_NORMAL

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

def filter_reading(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
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

def get_xyz_from_scan(metadata, scan):
    """
    The function "get_xyz_from_scan" takes in information and a scan, uses a client's XYZLut to convert
    the scan to XYZ coordinates, and returns the XYZ coordinates as a flattened list.
    
    Args:
      metadata: The "metadata" parameter is a variable that contains information about the scan. It is used to
    create an XYZLut object.
      scan: The `scan` parameter is a 3D array representing the scan data. It typically contains
    information about the position and intensity of points in a 3D space.
    
    Returns:
      a list of flattened arrays representing the XYZ values extracted from the given scan.
    """
    
    xyzlut = client.XYZLut(metadata)
    xyz = xyzlut(scan)
    return [c.flatten() for c in np.dsplit(xyz, 3)]

def get_xyz(source, host_input, port_input):
  """
  The function `get_xyz` retrieves x, y, z coordinates from a scan, filters the coordinates based on
  specified limits, serializes the data, and sends it to a specified host and port.
  
  Args:
    source: The source parameter is the source of the scans. It could be a file path, a database
  connection, or any other source that provides the scans.
    host_input: The `host_input` parameter is the host address where you want to send the serialized
  data. It should be a string representing the IP address or domain name of the host.
    port_input: The `port_input` parameter is the port number that you want to use for the network
  communication. It is the port on which the data will be sent to the specified host.
  """
  metadata = source.metadata
  scans = client.Scans(source)

  for scan in scans:
    if scan is not None:
      # Get x, y, z coordinates from the scan
      x, y, z = get_xyz_from_scan(metadata, scan)
      
      # Define your x, y, z coordinate limits for filtering
      x_min, x_max = -4, 4
      y_min, y_max = -4, 4
      z_min, z_max = -4, 4

      # Filter the coordinates
      x, y, z = filter_reading(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max)
      x = x.tolist()
      y = y.tolist()
      z = z.tolist()
      
      data = list(zip(x, y, z))
      serialized_data = json.dumps(data)
      send_function(serialized_data, host_input, port_input, measure_speed=measure_speed)

def calculate_speed(start_time, end_time, data_size):
  """
  The function calculates the speed in Mbps and the time taken in milliseconds to transfer a given
  data size.

  Args:
    start_time: The start time is the time when the data transfer started. It can be represented as a
  timestamp or a datetime object.
    end_time: The end_time parameter represents the time at which the data transfer ended.
    data_size: The `data_size` parameter represents the size of the data in bytes.

  Returns:
    the speed in Mbps (megabits per second) and the time taken in milliseconds.
  """
  time_taken = (end_time - start_time) * 1000  # Convert to milliseconds
  data_size_bits = data_size * 8
  speed_mbps = (data_size_bits / (time_taken / 1000)) / (10 ** 6)
  return speed_mbps, time_taken

def send_function(data, host, port, measure_speed=False, chunk_size=1024):
  """
  The `send_function` function sends data to a specified host and port using a TCP socket, and
  optionally measures the speed of the transmission.
  
  Args:
    data: The `data` parameter is the data that you want to send to the server. It can be any type of
  data, but it will be converted to a string and encoded as UTF-8 before sending it over the network.
    host: The `host` parameter is the IP address or hostname of the server you want to send the data
  to.
    port: The `port` parameter is the port number on the server to which you want to send the data. It
  is an integer value that specifies the communication endpoint on the server.
    measure_speed: The `measure_speed` parameter is a boolean flag that determines whether to measure
  the speed of the data transmission. If set to `True`, the function will measure the time it takes to
  send the data and calculate the speed in Mbps (megabits per second). If set to `False`, the.
  Defaults to False
  
  Returns:
    The function does not return any value.
  """
  
  try:
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  except socket.error as e:
    print(f"Failed to create socket. Error: {e}")
    return

  try:
    # Connect to the server
    s.connect((host, port))
  except socket.error as e:
    print(f"Failed to connect to {host}:{port}. Error: {e}")
    return

  try:
    # Convert data to bytes
    data_to_send = str(data).encode('utf-8')
    total_size = len(data_to_send)
    print(f'Trying to send data | Size {total_size} bytes')
    
    # Measure start time if measure_speed is True
    if measure_speed:
        start_time = time.time()
    
    # Send data in chunks
    start = 0
    end = chunk_size
    while start < total_size:
        chunk = data_to_send[start:end]
        s.sendall(chunk)
        start += chunk_size
        end += chunk_size
    
    # Measure end time and calculate speed if measure_speed is True
    if measure_speed:
        end_time = time.time()
        speed_mbps, time_taken_ms = calculate_speed(start_time, end_time, len(data_to_send))
        print(f"Sent packet to {host}:{port} | Size {len(data_to_send)} | Speed: {speed_mbps:.2f} Mbps | Time to send: {time_taken_ms:.2f} ms")
    else:
        print(f"Sent packet to {host}:{port} | Size {len(data_to_send)}")
      
  except socket.error as e:
    print(f"Failed to send data. Error: {e}")
      
  finally:
    # Close the socket
    s.close()

# ------ Take hostname, host, port, and measure_speed as input from user ----- #
"""
!Observation:
For LiDAR sensors running in a local network add the .local after the hostname
Ex: os-122308001777.local

For more information regarding the LiDAR network connection read the documentation:
https://static.ouster.dev/sensor-docs/image_route1/image_route3/networking_guide/networking_guide.html
"""

host_and_port = input("Enter the host IP (Server) and port in the format {host}:{port}: ")
hostname_input = input("Enter the hostname of the LiDAR: ")
measure_speed_input = input("Do you want to measure speed? (y/n): ")

host_input, port_input = host_and_port.split(":")
port_input = int(port_input)
measure_speed = True if measure_speed_input.lower() == 'y' else False

# ----------------------------- Running the code ----------------------------- #
source = configure_and_connect_sensor(hostname_input)

try:
  while True:
      get_xyz(source, host_input, port_input)
except KeyboardInterrupt:
  print("\nProgram terminated by user. Exiting...")
except Exception as e:
  print("\nProgram terminated due to an error. Exiting...")
  print("Error: " + str(e))
  
# ------------------------------------- - ------------------------------------ #