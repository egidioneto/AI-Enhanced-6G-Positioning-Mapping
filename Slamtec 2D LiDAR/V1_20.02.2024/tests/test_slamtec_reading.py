# Set the PYTHONPATH environment variable
import os, sys
sys.path.append(os.path.abspath("C:/Users/mathe/Documents/Projects/IC-6G/project"))

import numpy as np
from utils import CoordinatesHandler as ch
from utils import GraphHandler as gh

commonLocation = "Slamtec Lidar/Slamtec Samples/"
# samples = commonLocation + "readings_2.txt"
samples = commonLocation + "p_readings1.txt"

sample1 = commonLocation + "p_reading_1.txt"
sample2 = commonLocation + "p_reading_2.txt"
sample3 = commonLocation + "p_reading_3.txt"
sample4 = commonLocation + "random_reading_1.txt"
sample5 = commonLocation + "random_reading_2.txt"
sample6 = commonLocation + "random_reading_3.txt"
sample7 = commonLocation + "random_reading_4.txt"

# ----------------------------- Getting readings ----------------------------- #
# Get array of readings
print('Reading file...')
readings = ch.read_multiple_samples_slamtec(sample2)
print('Readings:', len(readings))

# Filtering readings by setting maximum range
readings = ch.filter_readings(readings, -2,2,-2,2) # square 4x4

# Fixed square size of 1x1m, that is, 16 squares (Default values)
squares = ch.split_graph() 
    
# Loop through each reading and only+ plot every xth reading
for i in range(0, len(readings), 10):
    
    # ------------ reading each theta and r, then transforming to x, y ----------- #
    theta, r = readings[i]
    x, y = ch.polar_to_cartesian(r, theta)

    # ----------------------------- Testing clusters ----------------------------- #
    # !BUG: bug quadrant 10
    clusters = ch.identify_clusters(x, y, threshold=1, min_points=30)
    print(ch.find_clusters_square_index(squares, clusters))

    # --------------------------------- Plotting --------------------------------- #
    gh.plot_2d_graph_with_squares(x, y, squares, window_size=9)


