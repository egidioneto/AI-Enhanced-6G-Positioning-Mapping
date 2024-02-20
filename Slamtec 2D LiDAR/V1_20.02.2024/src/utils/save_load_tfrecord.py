import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to remove tf warnings
import tensorflow as tf
import numpy as np

def save_xyz_to_tfrecord(filename, directory, x, y, z):
    """
    The function `save_to_tfrecord` saves data `x`, `y`, and `z` to a TFRecord file with the specified
    filename and directory.
    
    Args:
      filename: The filename parameter is the name of the TFRecord file that will be created.
      directory: The directory parameter is the path to the directory where you want to save the
    TFRecord file.
      x: The parameter `x` represents the data for feature x that you want to save to the TFRecord file.
    It should be a list or array of floating-point numbers.
      y: The parameter 'y' in the function 'save_to_tfrecord' represents the values of the 'y' variable
    that you want to save to the TFRecord file. It should be a list or array of floating-point numbers.
      z: The parameter `z` represents an array of floating-point values.
    """
    filepath = os.path.join(directory, f"{filename}.tfrecord")
    
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    
    writer = tf.io.TFRecordWriter(filepath)
    
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    feature_dict = {
        'x': float_feature(x),
        'y': float_feature(y),
        'z': float_feature(z),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    writer.write(example.SerializeToString())
    writer.close()

def load_xyz_from_tfrecord(filename):
    """
    The function `load_from_tfrecord` loads data from a TFRecord file and returns the parsed data as
    numpy arrays.
    
    Args:
      filename: The `filename` parameter is the name of the TFRecord file from which you want to load
    the data.
    
    Returns:
      The function `load_from_tfrecord` returns three NumPy arrays: `x`, `y`, and `z`.
    """
    feature_description = {
        'x': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.VarLenFeature(tf.float32),
        'z': tf.io.VarLenFeature(tf.float32),
    }
    
    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.sparse.to_dense(parsed_example['x'])
        y = tf.sparse.to_dense(parsed_example['y'])
        z = tf.sparse.to_dense(parsed_example['z'])
        return x, y, z

    dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = dataset.map(_parse_function)
    
    for x, y, z in parsed_dataset:
        return x.numpy(), y.numpy(), z.numpy()

def save_labels_to_tfrecord(filename, directory, list_to_be_saved):
    """
    The function `save_list_to_tfrecord` saves a list of lists to a TFRecord file by flattening the
    list, converting it to a numpy array, and writing it to the file along with the number of sub-lists.
    
    Args:
      filename: The filename parameter is the name of the file you want to save the data to. It should
    be a string without the file extension. For example, if you want to save the data to a file named
    "data.tfrecord", you would pass "data" as the filename parameter.
      directory: The `directory` parameter is the path to the directory where the TFRecord file will be
    saved.
      list_to_be_saved: The `list_to_be_saved` parameter is a list of lists that you want to save to a
    TFRecord file. Each sublist represents a set of data that you want to store.
    """
    filepath = os.path.join(directory, f"{filename}.tfrecord")
    
    # Flatten the list of lists and convert to numpy array
    flat_list = [item for sublist in list_to_be_saved for item in sublist]
    flat_array = np.array(flat_list, dtype=np.int64)
    
    # Save the number of sub-lists for later reconstruction
    num_sublists = len(list_to_be_saved)
    
    writer = tf.io.TFRecordWriter(filepath)
    
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    feature_dict = {
        'selected_quadrants': int64_feature(flat_array),
        'num_sublists': int64_feature([num_sublists])
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    writer.write(example.SerializeToString())
    writer.close()

def load_labels_from_tfrecord(filename):
    """
    The function `load_list_from_tfrecord` loads a list of lists from a TFRecord file.
    
    Args:
      filename: The filename parameter is the name of the TFRecord file from which you want to load the
    list.
    
    Returns:
      the parsed dataset, which is a list of lists representing the selected quadrants.
    """
    feature_description = {
        'selected_quadrants': tf.io.VarLenFeature(tf.int64),
        'num_sublists': tf.io.FixedLenFeature([], tf.int64),
    }
    
    @tf.function
    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        flat_array = tf.sparse.to_dense(parsed_example['selected_quadrants'])
        num_sublists = parsed_example['num_sublists']
        
        # Reconstruct the list of lists
        step = tf.cast(len(flat_array), tf.int64) // num_sublists
        selected_quadrants = tf.reshape(flat_array, [-1, step])
        
        return selected_quadrants

    dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = dataset.map(_parse_function)
    
    for selected_quadrants in parsed_dataset:
        return selected_quadrants

# ------------------------------- Usage example ------------------------------ #
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
z = [7.0, 8.0, 9.0]

save_xyz_to_tfrecord(filename='test_1', directory='ouster\\data\\xyz-data\\tfrecord', x=x, y=y, z=z)

x, y, z = load_xyz_from_tfrecord(filename='ouster\\data\\xyz-data\\tfrecord\\test_1.tfrecord')
print(x, y, z)
# ------------------------------------- - ------------------------------------ #