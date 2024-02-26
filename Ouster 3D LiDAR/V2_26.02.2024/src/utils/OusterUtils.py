import os
import numpy as np
from ouster import client
from itertools import islice
import subprocess

# ---------------------------------- Config ---------------------------------- #
def set_lidar_mode(hostname, lidar_mode):
    """
    Valid Values:
        "512x10"
        "1024x10" (default)
        "2048x10"
        "512x20"
        "1024x20"

        client.LidarMode.MODE_512x10
        client.LidarMode.MODE_1024x10
        client.LidarMode.MODE_2048x10
        client.LidarMode.MODE_512x20
        client.LidarMode.MODE_1024x20

    What It Does
        The lidar_mode parameter sets two key aspects of the sensor's operation:

        Horizontal Resolution: The first number (e.g., 512, 1024, 2048) indicates the number of data points the sensor will 
        collect in one full 360-degree rotation. A higher number means more data points and thus higher resolution.

        Rotation Rate: The second number (e.g., 10, 20) indicates the number of full 360-degree rotations the sensor will 
        complete per second. A higher number means the sensor will rotate faster.

    Impact on Performance
        Range: The effective range of the sensor increases by 15-20% for every halving of the number of points gathered. 
        For example, "512x10" will have a 15-20% longer range than "512x20".

        Data Density: Higher resolution (e.g., "2048x10") will provide more detailed data but may require more computational power to process.

        Update Rate: A higher rotation rate (e.g., "512x20") will provide more frequent updates but may reduce the effective range.

    To calculate the number of readings per second, you can use the formula:
        Readings/Second = Horizontal Resolution*Rotation Rate
    
        Example:
        For "512x10" mode: 512*10=5120 readings/second

    Subsampling or Averaging Data to X Readings/Second

        Subsampling Method
            1. Capture Data: Collect data for one second at the high rate.
            2. Select One Reading: From the collected data, select one reading. This could be the first reading, the last, or a random reading within that      one-second window.
            3. Repeat: Continue this process for each subsequent second.

        Averaging Method
            1. Capture Data: Collect data for one second at the high rate.
            2. Average Readings: Calculate the average of all readings taken within that one-second window.
            3. Use the Average: Use this average as your one reading for that second.
            4. Repeat: Continue this process for each subsequent second.
    """

    config = client.SensorConfig()
    config.lidar_mode = lidar_mode
    client.set_config(hostname, config, persist=True, udp_dest_auto=True)

# ------------------------------- Reading data ------------------------------- #

def get_xyz_from_scan(info, scan):
    """
    The function "get_xyz_from_scan" takes in information and a scan, uses a client's XYZLut to convert
    the scan to XYZ coordinates, and returns the XYZ coordinates as a flattened list.
    
    Args:
      info: The "info" parameter is a variable that contains information about the scan. It is used to
    create an XYZLut object.
      scan: The `scan` parameter is a 3D array representing the scan data. It typically contains
    information about the position and intensity of points in a 3D space.
    
    Returns:
      a list of flattened arrays representing the XYZ values extracted from the given scan.
    """
    
    xyzlut = client.XYZLut(info)
    xyz = xyzlut(scan)
    return [c.flatten() for c in np.dsplit(xyz, 3)]

def pcap_read_packets(
        source: client.PacketSource,
        metadata: client.SensorInfo,
        num: int = 0  # not used in this example
):
    
    """
    The function `pcap_read_packets` reads packets from a pcap file and prints information about Lidar
    and Imu packets.
    
    :param source: The `source` parameter is of type `client.PacketSource` and represents the source of
    the packets. It could be a pcap file or a live sensor stream
    :type source: client.PacketSource
    :param metadata: The `metadata` parameter is of type `client.SensorInfo` and it represents the
    information about the sensor that captured the packets. It may contain details such as the sensor's
    model, firmware version, serial number, and other relevant information. This parameter is not used
    in the `pcap_read
    :type metadata: client.SensorInfo
    :param num: The `num` parameter is an optional parameter that specifies the number of packets to
    read from the pcap file. However, in this example, it is not used, defaults to 0
    :type num: int (optional)
    """
    for packet in source:
        if isinstance(packet, client.LidarPacket):
            # Now we can process the LidarPacket. In this case, we access
            # the measurement ids, timestamps, and ranges
            measurement_ids = packet.measurement_id
            timestamps = packet.timestamp
            ranges = packet.field(client.ChanField.RANGE)
            print(f'  encoder counts = {measurement_ids.shape}')
            print(f'  timestamps = {timestamps.shape}')
            print(f'  ranges = {ranges.shape}')

        elif isinstance(packet, client.ImuPacket):
            # and access ImuPacket content
            print(f'  acceleration = {packet.accel}')
            print(f'  angular_velocity = {packet.angular_vel}')

def pcap_query_scan(source: client.PacketSource,
                    metadata: client.SensorInfo,
                    num: int = 0):
    """
    The function `pcap_query_scan` queries and prints the available fields and their corresponding data
    types in a LidarScan from a given pcap file.
    
    Args:
      source (client.PacketSource): The `source` parameter is a `PacketSource` object from the `client`
    module. It represents the source of the pcap file.
      metadata (client.SensorInfo): The `metadata` parameter is of type `client.SensorInfo` and it
    represents the associated sensor information for the `PacketSource` from the pcap file. It contains
    information about the sensor such as its model, serial number, firmware version, etc.
      num (int): The `num` parameter is used to specify the scan number in a given pcap file. It starts
    from 0, so if you want to query the first scan in the pcap file, you would pass `num=0`. Defaults to
    0
    """
    scans = iter(client.Scans(source))

    scan = next(scans)
    print("Available fields and corresponding dtype in LidarScan")
    for field in scan.fields:
        print('{0:15} {1}'.format(str(field), scan.field(field).dtype))

# ------------------------------ Processing data ----------------------------- #
def filter_readings_ouster(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Filter Ouster LIDAR readings based on x, y, z coordinates.

    Args:
        x (numpy.ndarray): Array of x-coordinates.
        y (numpy.ndarray): Array of y-coordinates.
        z (numpy.ndarray): Array of z-coordinates.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.
        z_min (float): Minimum z-coordinate.
        z_max (float): Maximum z-coordinate.

    Returns:
        tuple: Filtered x, y, z coordinates as numpy arrays.
    """
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return x[mask], y[mask], z[mask]

# --------------------------- Converting pcap data --------------------------- #
def pcap_to_las_each_frame(source: client.PacketSource,
                metadata: client.SensorInfo,
                num: int = 0,
                las_dir: str = ".",
                las_base: str = "las_out",
                las_ext: str = "las"):
    """
    The function `pcap_to_las` converts lidar scans from a pcap file to las files, with one las file per
    lidar scan.
    
    :param source: A `client.PacketSource` object that represents the source of the pcap file
    :type source: client.PacketSource
    :param metadata: The `metadata` parameter is an instance of `client.SensorInfo` which contains
    information about the sensor used to capture the lidar scans. It includes details such as the format
    of the lidar data, the UDP profile used, and other sensor-specific information
    :type metadata: client.SensorInfo
    :param num: The `num` parameter is an optional parameter that specifies the number of lidar scans to
    convert to LAS files. If `num` is not specified or set to 0, all lidar scans in the pcap file will
    be converted, defaults to 0
    :type num: int (optional)
    :param las_dir: The `las_dir` parameter is a string that specifies the directory where the LAS files
    will be saved. By default, it is set to the current directory (".") but you can provide a different
    directory path if you want the LAS files to be saved in a specific location, defaults to .
    :type las_dir: str (optional)
    :param las_base: The `las_base` parameter is a string that specifies the base name for the output
    LAS files. Each LAS file will be named using this base name followed by an index number. For
    example, if `las_base` is set to "las_out", the first LAS file will be named "las, defaults to
    las_out
    :type las_base: str (optional)
    :param las_ext: The `las_ext` parameter is a string that specifies the file extension for the output
    LAS files. By default, it is set to "las", defaults to las
    :type las_ext: str (optional)
    """

    if (metadata.format.udp_profile_lidar ==
            client.UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL):
        print("Note: You've selected to convert a dual returns pcap to LAS. "
              "Second returns are ignored in this conversion by this example "
              "for clarity reasons.  You can modify the code as needed by "
              "accessing it through Github or the SDK documentation.")

    from itertools import islice
    import laspy  # type: ignore

    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    if num:
        scans = islice(scans, num)

    for idx, scan in enumerate(scans):

        xyz = xyzlut(scan.field(client.ChanField.RANGE))

        las = laspy.create()
        las.x = xyz[:, :, 0].flatten()
        las.y = xyz[:, :, 1].flatten()
        las.z = xyz[:, :, 2].flatten()

        las_path = os.path.join(las_dir, f'{las_base}_{idx:06d}.{las_ext}')
        print(f'write frame #{idx} to file: {las_path}')

        las.write(las_path)

def pcap_to_las(source: client.PacketSource,
                            metadata: client.SensorInfo,
                            num: int = 0,
                            las_dir: str = ".",
                            las_file_name: str = "output.las"):
    """
    Modified function to convert all lidar scans from a pcap file to a single LAS file.
    """

    if (metadata.format.udp_profile_lidar ==
            client.UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL):
        print("Note: You've selected to convert a dual returns pcap to LAS. "
              "Second returns are ignored in this conversion.")

    import laspy  # type: ignore

    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    if num:
        scans = islice(scans, num)

    all_x = []
    all_y = []
    all_z = []

    for idx, scan in enumerate(scans):
        xyz = xyzlut(scan.field(client.ChanField.RANGE))
        all_x.extend(xyz[:, :, 0].flatten())
        all_y.extend(xyz[:, :, 1].flatten())
        all_z.extend(xyz[:, :, 2].flatten())

    las = laspy.create()
    las.x = all_x
    las.y = all_y
    las.z = all_z

    las_path = os.path.join(las_dir, las_file_name)
    print(f'Writing all frames to file: {las_path}')

    las.write(las_path)

def pcap_to_pcd(source: client.PacketSource,
                metadata: client.SensorInfo,
                num: int = 0,
                pcd_dir: str = ".",
                pcd_base: str = "pcd_out",
                pcd_ext: str = "pcd"):
    "Write scans from a pcap to pcd files (one per lidar scan)."

    if (metadata.format.udp_profile_lidar ==
            client.UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL):
        print("Note: You've selected to convert a dual returns pcap. Second "
              "returns are ignored in this conversion by this example "
              "for clarity reasons.  You can modify the code as needed by "
              "accessing it through github or the SDK documentation.")

    from itertools import islice
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError:
        print(
            "This example requires open3d, which may not be available on all "
            "platforms. Try running `pip3 install open3d` first.")
        exit(1)

    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)

    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    if num:
        scans = islice(scans, num)

    for idx, scan in enumerate(scans):

        xyz = xyzlut(scan.field(client.ChanField.RANGE))

        pcd = o3d.geometry.PointCloud()  # type: ignore

        pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1,
                                                            3))  # type: ignore

        pcd_path = os.path.join(pcd_dir, f'{pcd_base}_{idx:06d}.{pcd_ext}')
        print(f'write frame #{idx} to file: {pcd_path}')

        o3d.io.write_point_cloud(pcd_path, pcd)  # type: ignore

def pcap_to_ply(source: client.PacketSource,
                metadata: client.SensorInfo,
                num: int = 0,
                ply_dir: str = ".",
                ply_base: str = "ply_out",
                ply_ext: str = "ply"):
    "Write scans from a pcap to ply files (one per lidar scan)."

    # Don't need to print warning about dual returns since this leverages pcap_to_pcd

    # We are reusing the same Open3d File IO function to write the PLY file out
    pcap_to_pcd(source,
                metadata,
                num=num,
                pcd_dir=ply_dir,
                pcd_base=ply_base,
                pcd_ext=ply_ext)

def pcap_to_csv(pcap_file: str, out_csv_path: str):
    """
    The function `pcap_to_csv` converts a pcap file to a csv file using the `ouster-cli` command line
    tool.
    
    Args:
      pcap_file (str): The `pcap_file` parameter is a string that represents the path to the pcap file
    that you want to convert to CSV format.
      out_csv_path (str): The `out_csv` parameter is a string that represents the path and filename of the
    output CSV file. This is the file where the converted data from the pcap file will be saved.
    """
    try:
        subprocess.run(["ouster-cli", "source", pcap_file, "convert", out_csv_path], check=True)
    except subprocess.CalledProcessError:
        print("Command failed. Make sure ouster-cli is installed and the input files are correct.")