import cv2
import numpy as np
from contextlib import closing
from ouster import client
import time

def stream_live_data(hostname, lidar_port=7502, frequency=0):
    """
    Stream live data from a configured sensor and display it as a black and white moving image.
    
    Parameters:
    - hostname (str): The hostname of the sensor.
    - lidar_port (int): The port number for the LiDAR sensor.
    - frequency (int): Time interval between frames in seconds.
    
    To exit the visualization, press ESC.
    """
    # Establish sensor connection
    with closing(client.Scans.stream(hostname, lidar_port, complete=False)) as stream:
        show = True
        while show:
            for scan in stream:
                # Uncomment the next line if you'd like to see the frame id printed
                # print("frame id: {} ".format(scan.frame_id))
                
                # Get reflectivity data and destagger it
                reflectivity = client.destagger(stream.metadata, scan.field(client.ChanField.REFLECTIVITY))
                
                # Scale the reflectivity data
                reflectivity = (reflectivity / np.max(reflectivity) * 255).astype(np.uint8)
                
                # Display the scaled reflectivity
                cv2.imshow("scaled reflectivity", reflectivity)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Exit the loop if ESC key is pressed
                if key == 27:
                    show = False
                    break
                
                # Adding sleep time to reduce frequency
                time.sleep(frequency)

# ----------------------------------- Usage ---------------------------------- #
stream_live_data('os-122308001777', frequency=0)  # Replace it with the actual hostname of your sensor