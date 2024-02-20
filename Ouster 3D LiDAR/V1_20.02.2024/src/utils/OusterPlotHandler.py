import ouster.client as client
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# --------------------------------- Functions -------------------------------- #
def plot_3d(x, y, z):
    """Plotting points 3d - Specific setting for plotting"""
    ax = plt.axes(projection='3d')
    r = 10
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r/2, r/2])
    plt.axis('off')
    z_col = np.minimum(np.absolute(z), 5)
    ax.scatter(x, y, z, c=z_col, s=0.2)
    plt.show()

# ------------------------ Using metadata from ouster ------------------------ #
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

def visualize_with_matplotlib_2d(range_img):
    """
    Visualizes a 2D range image using Matplotlib.

    Args:
        range_img (numpy.ndarray): The range image to visualize.

    Usage:
        visualize_with_matplotlib_2d(range_img)
    """
    plt.imshow(range_img[:, 640:1024], resample=False)
    plt.axis('off')
    plt.show()

def visualize_with_matplotlib_3d(info, scan):
    """
    Visualizes a 3D point cloud using Matplotlib.

    Args:
        info (client.SensorInfo): The sensor information.
        scan (client.LidarScan): The Lidar scan data.

    Usage:
        visualize_with_matplotlib_3d(info, scan)
    """
    xyzlut = client.XYZLut(info)
    xyz = xyzlut(scan)
    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
    ax = plt.axes(projection='3d')
    r = 10
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r/2, r/2])
    plt.axis('off')
    z_col = np.minimum(np.absolute(z), 5)
    ax.scatter(x, y, z, c=z_col, s=0.2)
    plt.show()

# ------------------------------------- - ------------------------------------ #
visualize_with_simple_viz(sensor_hostname="os-122308001777")


"""def visualize_with_open3d(metadata, scan):
    # !Supported Python versions: 3.7 3.8 3.9 3.10
    import open3d as o3d
    
    # Compute point cloud using client.SensorInfo and client.LidarScan
    xyz = client.XYZLut(metadata)(scan)

    # Create point cloud and coordinate axes geometries
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.reshape((-1, 3))))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)

    # Add more code as needed for specific visualization"""