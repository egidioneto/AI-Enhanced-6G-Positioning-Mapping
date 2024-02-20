from ouster import client
from ouster import pcap
from utils import OusterUtils as outils

# ------------------------- Paths to the sample data ------------------------- #
common_path = 'C:\\Users\\mathe\\Documents\\Projects\\IC-6G\\project\\ouster\\data\\pcap-recordings\\'
pcap_path = common_path + 'recording-20230905_155620.pcap'
metadata_path = common_path + 'recording-20230905_155620.json'

# Importing the client module and loading the sensor information
with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

# Importing the pcap module and reading the captured UDP data
source = pcap.Pcap(pcap_path, metadata)

# ------------------------------- Testing query ------------------------------ #
outils.pcap_query_scan(source, metadata, 0)

# -------------------------------- pcap to csv ------------------------------- #
outils.pcap_to_csv(pcap_path, 
                   "C:/Users/mathe/Documents/Projects/IC - 6G/Python - Project/Ouster Lidar/CSV Recordings/recording-20230905_155620.csv")

# -------------------------------- pcap to las ------------------------------- #
num = 0  # Set the desired scan number from the pcap file (e.g., 0 for the first scan)
las_dir = 'Ouster Lidar\LAS Recordings'  # Set the directory where you want to save the LAS files
las_name = 'recording-20230905_155620.las'  # Set the base name for the LAS files

outils.pcap_to_las(source, metadata, num, las_dir, las_name)
