import subprocess

def visualize_with_simple_viz(sensor_hostname=None, pcap_path=None):
    """
    Visualizes Ouster sensor data using Ouster's command-line utility.

    Args:
        sensor_hostname (str): The hostname or IP of the sensor.
        pcap_path (str): The path to the pcap file.

    Usage:
        # Using a sensor
        visualize_with_simple_viz(sensor_hostname="os-122308001777")

        # Using a pcap file
        visualize_with_simple_viz(pcap_path="path/to/pcap.pcap")
        
    !If it doesn't work properly, try using the command directly on the terminal
    
    ouster-cli source 'pcap_path' viz
    ouster-cli source os-122308001777 viz
    """
    if sensor_hostname:
        source = sensor_hostname
    elif pcap_path:
        source = pcap_path
    else:
        print("Provide either sensor hostname or pcap path.")
        return

    command = ["ouster-cli", "source", source, "viz"]
    subprocess.run(command)
    
visualize_with_simple_viz(sensor_hostname="os-122308001777")