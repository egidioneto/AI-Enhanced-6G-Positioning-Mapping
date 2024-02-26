import socket
import time

# ---------------------------------------------------------------------------- #
#               This code is the client, that is, the data sender              #
# ---------------------------------------------------------------------------- #

# !------------------------- This is a test script -------------------------! #

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

def send_function(data, host, port, measure_speed=False):
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
        
        # Measure start time if measure_speed is True
        if measure_speed:
            start_time = time.time()
        
        # Send data
        s.sendall(data_to_send)
        
        # Measure end time and calculate speed if measure_speed is True
        if measure_speed:
            end_time = time.time()
            speed_mbps, time_taken_ms = calculate_speed(start_time, end_time, len(data_to_send))
            print(f"Sent packet to {host}:{port} at {speed_mbps:.2f} Mbps, Time to send: {time_taken_ms:.2f} ms")
        else:
            print(f"Sent packet to {host}:{port}")
        
    except socket.error as e:
        print(f"Failed to send data. Error: {e}")
        
    finally:
        # Close the socket
        s.close()

# ----------------------------- Running the code ----------------------------- #
# Take host, port, and measure_speed as input from user
host_and_port = input("Enter the host IP (Server) and port  in the format {host}:{port}: ")
measure_speed_input = input("Do you want to measure speed? (y/n): ")

host_input, port_input = host_and_port.split(":")
port_input = int(port_input)
measure_speed = True if measure_speed_input.lower() == 'y' else False

while True:
    send_function("Hello, world!", host_input, port_input, measure_speed=measure_speed)
