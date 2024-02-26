from ouster import client
from ouster import pcap
from contextlib import closing
from more_itertools import nth
from utils import OusterPlotHandler as oph
from utils import OusterUtils as outils

# ------------------------- Paths to the sample data ------------------------- #
# pcap_path = 'D:\\IC - 6G - HD\\Ouster Samples\\Example 1\\pcap1.pcap'
# metadata_path = 'D:\\IC - 6G - HD\\Ouster Samples\\Example 1\\json1.json'

pcap_path = 'data\\pcap-recordings\\recording-20230905_155620.pcap'
metadata_path = 'data\\pcap-recordings\\recording-20230905_155620.json'

# Importing the client module and loading the sensor information
with open(metadata_path, 'r') as f:
    info = client.SensorInfo(f.read())

# Importing the pcap module and reading the captured UDP data
source = pcap.Pcap(pcap_path, info)

# ---------------------- Proceed to Visualizations in 3D --------------------- #
with closing(client.Scans(source)) as scans:
    scan = nth(scans, 50) # !assuming that you want to visualize the 50th frame of data

range_field = scan.field(client.ChanField.RANGE)
range_img = client.destagger(info, range_field)

# Using Ouster's simple-viz with pcap and metadata file
oph.visualize_with_simple_viz(pcap_path=pcap_path)
# oph.visualize_with_simple_viz(sensor_hostname="os-122308001777")

# 2D Visualization with Matplotlib
oph.visualize_with_matplotlib_2d(range_img)

# 3D Visualization with Matplotlib
oph.visualize_with_matplotlib_3d(info, scan)

# ---------------------- Filtering the LIDAR Readings ------------------------ #
# Get x, y, z coordinates from the scan
x, y, z = outils.get_xyz_from_scan(info, scan)

# Define your x, y, z coordinate limits for filtering
x_min, x_max = -4, 4
y_min, y_max = -4, 4
z_min, z_max = -4, 4

# Filter the coordinates
filtered_x, filtered_y, filtered_z = outils.filter_readings_ouster(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max)

# Plot the filtered coordinates
oph.plot_3d(filtered_x, filtered_y, filtered_z)