from ouster import pcap
from more_itertools import time_limited
from contextlib import closing
import ouster.client as client
from datetime import datetime
import os

def record_and_save_sensor_data(hostname, lidar_port, imu_port, n_seconds, save_folder):
    '''The function `record_and_save_sensor_data` records sensor data from a specified hostname and ports
    for a specified duration and saves the data in a specified folder.
    
    Parameters
    ----------
    hostname
        The hostname is the IP address or hostname of the sensor device you want to connect to. This is
    where the sensor data is being streamed from.
    lidar_port
        The `lidar_port` parameter is the port number used to connect to the LiDAR sensor. It is the
    communication channel through which the LiDAR sensor sends data.
    imu_port
        The `imu_port` parameter is the port number on which the IMU (Inertial Measurement Unit) sensor is
    connected. The IMU sensor is responsible for measuring the orientation, velocity, and gravitational
    forces acting on an object.
    n_seconds
        The `n_seconds` parameter specifies the duration in seconds for which the sensor data should be
    recorded.
    save_folder
        The `save_folder` parameter is the path to the folder where the recorded sensor data will be saved.
    This folder will be created if it doesn't exist.
    
    '''
    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with closing(client.Sensor(hostname, lidar_port, imu_port, buf_size=640)) as source:
        # Generate filename
        time_part = datetime.now().strftime("%d_%m_%Y-%H_%M")

        meta = source.metadata
        fname_base = f"metadata-{time_part}"

        # Save metadata
        metadata_path = os.path.join(save_folder, f"{fname_base}.json")
        print(f"Saving sensor metadata to: {metadata_path}")
        source.write_metadata(metadata_path)

        # Record and save pcap data
        pcap_path = os.path.join(save_folder, f"{fname_base}.pcap")
        print(f"Writing to: {pcap_path} (Ctrl-C to stop early)")
        source_it = time_limited(n_seconds, source)
        n_packets = pcap.record(source_it, pcap_path)

        print(f"Captured {n_packets} packets")

        """
        Usage example:

        record_and_save_sensor_data(hostname="your_sensor_hostname", 
                            lidar_port=your_lidar_port, 
                            imu_port=your_imu_port, 
                            n_seconds=10, 
                            save_folder="your_save_folder")
        """

# ----------------------------------- Usage ---------------------------------- #
record_and_save_sensor_data(hostname="os-122308001777", 
                            lidar_port=7502, 
                            imu_port=7503, 
                            n_seconds=10, 
                            save_folder="data\pcap-recordings")