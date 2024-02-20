import os
import tensorflow as tf
import csv
import numpy as np
from plyfile import PlyData, PlyElement
import laspy

# ------------------------------ Save Functions ------------------------------ #
# Save to CSV
def save_to_csv(x, y, z, filename, directory):
    """
    The function `save_to_csv` saves three lists `x`, `y`, and `z` as columns in a CSV file with the
    specified `filename` and `directory`.
    
    Args:
      x: A list of values for the x-axis.
      y: The parameter "y" is a list or array containing the values for the "y" variable.
      z: The parameter "z" represents a list of values that you want to save to the CSV file.
      filename: The filename parameter is the name of the CSV file you want to save the data to.
      directory: The directory parameter is the path to the directory where you want to save the CSV
    file.
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(x, y, z):
            csvwriter.writerow([xi, yi, zi])

# Save to .npy
def save_to_npy(x, y, z, filename, directory):
    """
    The function saves three arrays (x, y, and z) to a .npy file with a specified filename and
    directory.
    
    Args:
      x: The parameter `x` is a variable that represents some data that you want to save to a .npy file.
      y: The parameter "y" is a variable that represents some data that you want to save to a .npy file.
      z: The parameter "z" is a variable that represents some data that you want to save to a .npy file.
      filename: The name of the file to save the data to.
      directory: The directory parameter is the path to the directory where you want to save the .npy
    file.
    """
    filepath = os.path.join(directory, filename)
    np.save(filepath, {'x': x, 'y': y, 'z': z})

# Save to PLY
def save_to_ply(x, y, z, filename, directory):
    """
    The function `save_to_ply` saves a set of 3D coordinates to a PLY file in a specified directory.
    
    Args:
      x: A list or array of x-coordinates of the vertices.
      y: The parameter "y" represents the y-coordinates of the vertices in the 3D space.
      z: The parameter `z` represents the z-coordinates of the vertices.
      filename: The filename parameter is the name of the file you want to save the data to. It should
    include the file extension, such as ".ply".
      directory: The directory parameter is the path to the directory where you want to save the PLY
    file.
    """
    filepath = os.path.join(directory, filename)
    vertex = np.array([(xi, yi, zi) for xi, yi, zi in zip(x, y, z)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filepath)

# Save to LAS
def save_to_las(x, y, z, filename, directory):
    """
    The function `save_to_las` saves x, y, and z coordinates to a LAS file with the specified filename
    and directory.
    
    Args:
      x: The x-coordinates of the points to be saved in the LAS file.
      y: The parameter "y" represents the y-coordinate values of the points to be saved to the LAS file.
      z: The parameter "z" represents the array or list of z-coordinates that you want to save to the
    LAS file.
      filename: The filename parameter is the name of the LAS file that you want to save. It should
    include the file extension ".las".
      directory: The directory parameter is the path to the directory where you want to save the LAS
    file.
    """
    filepath = os.path.join(directory, filename)
    hdr = laspy.header.Header()
    outfile = laspy.file.File(filepath, mode="w", header=hdr)
    outfile.x = x
    outfile.y = y
    outfile.z = z
    outfile.close()
    
# ------------------------------ Load Functions ------------------------------ #
# Load from CSV
def load_from_csv(filename):
  """
  The function `load_from_csv` reads data from a CSV file and returns three arrays containing the
  values from the first, second, and third columns respectively.
  
  Args:
    filename: The filename parameter is the name of the CSV file that you want to load data from.
  
  Returns:
    three numpy arrays: x, y, and z.
  """
  x, y, z = [], [], []
  with open(filename, 'r') as csvfile:
      csvreader = csv.reader(csvfile)
      next(csvreader)  # Skip header
      for row in csvreader:
          x.append(float(row[0]))
          y.append(float(row[1]))
          z.append(float(row[2]))
  return np.array(x), np.array(y), np.array(z)

# Load from .npy
def load_from_npy(filename):
  """
  The function `load_from_npy` loads data from a .npy file and returns three arrays.
  
  Args:
    filename: The filename parameter is the name of the .npy file from which you want to load the
  data.
  
  Returns:
    three numpy arrays: `x`, `y`, and `z`.
  """
  data = np.load(filename, allow_pickle=True).item()
  return np.array(data['x']), np.array(data['y']), np.array(data['z'])

# Load from PLY
def load_from_ply(filename):
  """
  The function `load_from_ply` reads vertex data from a PLY file and returns arrays of x, y, and z
  coordinates.
  
  Args:
    filename: The filename parameter is a string that represents the name or path of the PLY file that
  you want to load.
  
  Returns:
    three numpy arrays: x, y, and z.
  """
  plydata = PlyData.read(filename)
  vertex = plydata['vertex']
  x = vertex['x']
  y = vertex['y']
  z = vertex['z']
  return np.array(x), np.array(y), np.array(z)

# Load from LAS
def load_from_las(filename):
  """
  The function `load_from_las` loads x, y, and z coordinates from a LAS file and returns them as numpy
  arrays.
  
  Args:
    filename: The filename parameter is a string that represents the name or path of the LAS file that
  you want to load.
  
  Returns:
    three numpy arrays: x, y, and z.
  """
  infile = laspy.file.File(filename, mode="r")
  x = infile.x
  y = infile.y
  z = infile.z
  return np.array(x), np.array(y), np.array(z)