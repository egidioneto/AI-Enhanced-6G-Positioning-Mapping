from ouster import client

def get_lidar_data(hostname):
    """
    This function connects to an Ouster lidar sensor using its hostname,
    configures the sensor, and then grabs a set of data from it.
    
    Parameters:
    - hostname (str): The hostname or IP address of the Ouster lidar sensor.
    
    Returns:
    - (info, source): The data grabbed from the sensor.
    """
    
    # Initialize the SensorConfig object from the Ouster client module.
    # This object will hold the configuration settings for the sensor.
    config = client.SensorConfig()
    
    # Set the UDP port for lidar data. The default is usually 7502.
    config.udp_port_lidar = 7502
    
    # Set the UDP port for IMU data. The default is usually 7503.
    config.udp_port_imu = 7503
    
    # Set the operating mode of the sensor. Here, we set it to normal operation.
    config.operating_mode = client.OperatingMode.OPERATING_NORMAL
    
    # Apply the configuration settings to the sensor.
    # The 'persist=True' argument means that the settings will be saved on the sensor.
    # The 'udp_dest_auto=True' argument allows the SDK to automatically set the UDP destination.
    client.set_config(hostname, config, persist=True, udp_dest_auto=True)
    
    # Create a PacketSource from the sensor using its hostname and the UDP ports for lidar and IMU data.
    # This object acts as a source of data packets from the sensor.
    source = client.Sensor(hostname, 7502, 7503)
    
    # Retrieve the metadata. This contains information like sensor settings, calibration data, etc.
    info = source.metadata
    
    # Return the grabbed data
    return (info, source)

# ----------------------------------- Usage ---------------------------------- #
# Grabbing sensor source and metadata
metadata, source = get_lidar_data('os-122308001777') # Replace it with the actual hostname of your sensor
print("Grabbed data successfully!")