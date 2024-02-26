import socket
import psutil
import ast
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
#              This code is the server, that is, the data receiver             #
# ---------------------------------------------------------------------------- #

"""
# ------------------------------- Documentation ------------------------------ #

!1) Make sure to allow the Python application through your firewall.
!2) Run the server code before running the client code.!
3) Both the server and client should be on the same network for this to work.

!Steps to Set Up Server on Your Computer:
    1. Find IP Address
        Windows: Open Command Prompt and run ipconfig. Look for "IPv4 Address" under your active network connection.
        macOS/Linux: Open Terminal and run ifconfig or ip a. Look for "inet" under your active network interface.

    2. Run Server Code
        Copy the server code into a Python file (e.g., server.py).
        Replace the host variable with the IP address you found.
        Run the Python file to start the server.

    3. Firewall Settings
        Windows: Allow the Python application through the Windows Firewall when prompted.
        macOS/Linux: You might need to adjust your firewall settings to allow incoming connections on the chosen port.
        
!Choosing a Port Number:
    The port number is an arbitrary number you choose to identify a specific application-level service on your server. Here are some guidelines:

    1. Well-Known Ports: 0-1023 (Reserved for well-known services like HTTP, FTP, etc. Avoid these unless you're running a standard service.)
    2. Registered Ports: 1024-49151 (Used by software applications. Some may be taken by other services.)
    3. Dynamic/Private Ports: 49152-65535 (Typically safe to use for custom applications.)
        
    !Steps to Choose a Port:
        1. Pick a Number: Choose a number between 1024 and 65535 that you'll remember.
        2. Check Availability: Make sure no other service is using the port you've chosen.
            !Check Port Availability: 
                Windows: Open Command Prompt and run netstat -an | find "PORT_NUMBER".
                macOS/Linux: Open Terminal and run netstat -an | grep PORT_NUMBER.
        3. Replace PORT_NUMBER with the port number you've chosen. If nothing comes up, the port is likely available.
"""

# ------------------------ Handling KeyboardInterrupts ----------------------- #
import signal
import sys
def signal_handler(signum, frame):
    """
    The function `signal_handler` is used to handle a signal and print a message before exiting the
    program.
    
    Args:
      signum: The signum parameter represents the signal number that triggered the signal handler.
    Signals are used in operating systems to communicate events or interrupts to processes. Examples of
    signals include SIGINT (interrupt signal) and SIGTERM (termination signal).
      frame: The frame parameter represents the current stack frame at the time the signal was received.
    It contains information about the execution context, such as the current line number and local
    variables.
    """
    print("\nShutting down connection...")
    print("Did you allow the connection through the firewall?")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ------------------------------ Util functions ------------------------------ #
def plot_3d(x, y, z):
    """
    The function `plot_3d` plots a 3D scatter plot of points with given x, y, and z coordinates.
    
    Args:
      x: The x parameter represents the x-coordinates of the points in the 3D plot. It is a
    1-dimensional array or list of values.
      y: The parameter `y` represents the y-coordinates of the points in the 3D plot.
      z: The parameter `z` represents the z-coordinates of the points in the 3D plot. It determines the
    vertical position of each point in the plot.
    """
    ax = plt.axes(projection='3d')
    r = np.max([np.abs(np.min(x)), np.abs(np.max(x)), 
                np.abs(np.min(y)), np.abs(np.max(y)), 
                np.abs(np.min(z)), np.abs(np.max(z))])  # Adjust r based on the data
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r/2, r/2])
    plt.axis('off')
    z_col = np.minimum(np.absolute(z), 5)
    ax.scatter(x, y, z, c=z_col, s=0.2)
    ax.view_init(30, 30)  # Adjusted view
    plt.show()
    
def filter_reading(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    The function filters readings based on specified minimum and maximum values for x, y, and z.
    
    Args:
      x: An array of x-coordinates of readings.
      y: The parameter "y" is a list or array containing the values of the y-coordinate for each data
    point.
      z: The parameter "z" represents an array of readings or values for the z-axis.
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

# ----------------------------- Server functions ----------------------------- #
def get_active_network_interface_ip_and_name():
    """
    The function `get_active_network_interface_ip_and_name` returns the IP address and the name of the active network
    interface prioritizing WiFi.
    
    Returns:
      Tuple containing the IP address and the name of the first active network interface, or (None, None) if there are no active interfaces.
    """
    wifi_interface = None
    other_interface = None
    
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                if psutil.net_if_stats()[interface].isup:  # Check if interface is active
                    if 'wlan' in interface.lower() or 'wi-fi' in interface.lower():  # Check for WiFi interface
                        wifi_interface = (addr.address, interface)
                    else:
                        other_interface = (addr.address, interface)

    if wifi_interface:
        return wifi_interface  # Return WiFi IP and interface name if available
    elif other_interface:
        return other_interface  # Return other active interface if WiFi is not available

    return None, None  # Return None if no active interface found

def receive_data(chunk_size=1024):
    """
    The function `receive_data` creates a socket, binds it to an IP address and port, listens for
    incoming connections, and receives data from the client.
    
    Args:
      chunk_size: The `chunk_size` parameter is used to specify the size of each chunk of data that is
    received from the client. It determines how much data is received at a time before processing it.
    The default value is 1024 bytes. Defaults to 1024
    
    Returns:
      The function does not explicitly return any value.
    """
    print("Please ensure that your firewall settings allow incoming connections to the server.\n")
    
    # Create a socket object
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print(f"Failed to create socket. Error: {e}")
        return

    # Automatically get the active network interface's IP address
    host, interface_name = get_active_network_interface_ip_and_name()

    if host is None:
        print("No active network interface found.")
        user_input = input("Would you like to enter the IP manually? (y/n): ")
        if user_input.lower() == 'y':
            host = input("Enter the IP address: ")
        else:
            return
    else:
        print(f"Detected IP address: {host} from interface: {interface_name}")
        user_input = input("Would you like to use this IP address? (y/n): ")
        if user_input.lower() == 'n':
            user_input = input("Would you like to select an IP from the list of active interfaces or enter manually? (list/manual): ")
            if user_input.lower() == 'list':
                available_ips_and_interfaces = [(addr.address, iface) for iface, addrs in psutil.net_if_addrs().items() for addr in addrs if addr.family == socket.AF_INET]
                print("Available IP addresses and their interfaces:")
                for i, (ip, iface) in enumerate(available_ips_and_interfaces):
                    print(f"{i}. {ip} ({iface})")
                selected_index = int(input("Enter the index of the IP you'd like to use: "))
                host, _ = available_ips_and_interfaces[selected_index]
            else:
                host = input("Enter the IP address manually: ")

    # Let the OS pick an available port
    port = 0
    user_input = input("Would you like to enter the port manually? (y/n): ")
    if user_input.lower() == 'y':
        port = int(input("Enter the port number: "))

    # Bind the socket to the automatically obtained IP address and a random port
    try:
        s.bind((host, port))
    except socket.error as e:
        print(f"Failed to bind socket. Error: {e}")
        return

    # Retrieve the port number
    port = s.getsockname()[1]

    # Listen for incoming connections
    try:
        s.listen(5)
    except socket.error as e:
        print(f"Failed to listen on socket. Error: {e}")
        return

    print(f"Server running on {host}:{port}") 

    while True:
        # Accept a connection
        try:
            c, addr = s.accept()
        except socket.error as e:
            print(f"Failed to accept connection. Error: {e}")
            continue

        # Receive data from the client
        try:
            received_data = b''
            while True:
                chunk = c.recv(chunk_size)
                if not chunk:
                    break
                received_data += chunk
            
            received_data = received_data.decode('utf-8')
            size = len(received_data)
            received_data = ast.literal_eval(received_data)
            print(f"Received data, type {type(received_data)} | size {size} bytes")
            
            # ---------------------------- Using received data --------------------------- #
            # Get x, y, z coordinates from the received data
            x = [point[0] for point in received_data]
            y = [point[1] for point in received_data]
            z = [point[2] for point in received_data]

            # Plot the filtered coordinates
            # plot_3d(x, y, z)
            # ------------------------------------- - ------------------------------------ #
            
        except socket.error as e:
            print(f"Failed to receive data. Error: {e}")
            c.close()
            continue
        
        # Close the connection
        c.close()
     
# ------------------------------- Running code ------------------------------- #
receive_data()