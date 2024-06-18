import os
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout # type: ignore
from keras.models import Sequential, load_model # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------- Auxiliary functions --------------------------- #
def load_results_from_txt(filename="results.txt"):
    """
    Loads the selected quadrants from a text file and converts them back to a list of lists.

    Args:
        filename (str): The name of the file to read from.

    Returns:
        list: A list of lists, where each sublist contains integers representing selected quadrants.
    """
    selected_quadrants = []
    with open(filename, "r") as f:
        for line in f:
            # Convert the line to a list of integers and append to selected_quadrants
            quadrant_list = list(map(int, line.strip().split(",")))
            selected_quadrants.append(quadrant_list)
    return selected_quadrants


def read_readings_from_file(filename, location):
    """
    The function `read_from_file` reads data from a file and returns it as a list of sublists, where
    each sublist contains pairs of floating-point numbers.

    Args:
      filename: The filename parameter is the name of the file you want to read from. It should include
    the file extension (e.g., "data.txt").
      location: The `location` parameter is the directory where the file is located. It should be a
    string representing the path to the directory.

    Returns:
      a list of sublists, where each sublist contains tuples of two floating-point numbers.
    """
    filepath = os.path.join(location, filename)
    data = []
    with open(filepath, "r") as file:
        sublist = []
        for line in file:
            if line.strip() == "":
                if sublist:
                    data.append(sublist)
                    sublist = []
            else:
                parts = line.split()
                sublist.append((float(parts[0]), float(parts[1])))
        if sublist:  # add the last sublist if file does not end with a newline
            data.append(sublist)
    return data


def shuffle_readings(readings, labels):
    """
    Shuffles the readings and their corresponding labels.

    Args:
        readings: The `readings` parameter is a list of sublists, where each sublist contains tuples
    representing x and y coordinates of readings. Each sublist represents a set of readings.
        labels: The `labels` parameter is a list of corresponding labels.

    Returns:
        tuple: A tuple containing two lists: shuffled readings and shuffled labels.
    """
    combined = list(zip(readings, labels))
    np.random.shuffle(combined)
    shuffled_readings, shuffled_labels = zip(*combined)
    
    # Convert the zipped lists back to their original types
    shuffled_readings = list(map(list, shuffled_readings))
    shuffled_labels = list(map(list, shuffled_labels))
    
    return shuffled_readings, shuffled_labels


def evaluate_model_with_metrics(model, X_test, y_test, filename="results"):
    """
    Evaluates the model and saves the metrics and confusion matrices to files.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: Test labels.
        filename: Base filename for saving results.
    """
    # Evaluate the model on the test data.
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

    # The accuracy is the proportion of correct predictions over total predictions.
    print("Test accuracy:", accuracy)

    # This is what the model aims to minimize during training.
    print("Test loss:", loss)

    # Predict the probabilities of each class for the test data.
    y_pred_prob = model.predict(X_test)

    # Convert the predicted probabilities into actual class predictions by setting a threshold.
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Flatten the arrays to treat all label predictions as binary
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate the individual components of the confusion matrix
    TP = ((y_pred_flat == 1) & (y_test_flat == 1)).sum()
    TN = ((y_pred_flat == 0) & (y_test_flat == 0)).sum()
    FP = ((y_pred_flat == 1) & (y_test_flat == 0)).sum()
    FN = ((y_pred_flat == 0) & (y_test_flat == 1)).sum()

    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # Calculate Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Calculate F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("\n# ---------------------------------- Results --------------------------------- #")
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Save results to a text file
    with open(f"{filename}.txt", "w") as file:
        file.write(f"Precision: {precision}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

    # Create a 2x2 confusion matrix
    general_cm = np.array([[TP, FP], [FN, TN]])

    # Plot the confusion matrix without colors
    plt.figure(figsize=(8, 6))
    sns.heatmap(general_cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
    plt.ylabel("Predicted Label", fontsize=14)
    plt.xlabel("Real Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)

    # Define class labels and their positions
    class_labels = ["Positive", "Negative"]
    plt.xticks(ticks=np.arange(2) + 0.5, labels=class_labels, ha="center")
    plt.yticks(ticks=np.arange(2) + 0.5, labels=class_labels, va="center", rotation=0)

    plt.savefig(f"{filename}.png")
    plt.close()


def process_readings(readings, squares, filename):
    """
    The function takes a list of readings and a list of squares, and returns a matrix where each row
    represents a reading and each column represents a square, with a value of 1 indicating that the
    reading falls within the corresponding square.

    Args:
      readings: The `readings` parameter is a list of sublists, where each sublist contains tuples
    representing x and y coordinates of readings. Each sublist represents a set of readings.
      squares: The `squares` parameter is a list of tuples, where each tuple represents the coordinates
    of a square. Each tuple contains four values: `x1`, `x2`, `y1`, `y2`. These values define the
    boundaries of the square in the x and y dimensions.

    Returns:
      a 2D numpy array called `processed_readings`.
    """
    processed_readings = np.zeros((len(readings), len(squares)))

    for i, sublist in enumerate(readings):
        for reading in sublist:
            x, y = reading  # Unpack the tuple directly
            for j, (x1, x2, y1, y2) in enumerate(squares):
                if x1 <= x < x2 and y1 <= y < y2:
                    processed_readings[i][j] = 1
                    break
                
    # Save the processed readings to a file
    np.savetxt(f"{filename}.csv", processed_readings, delimiter=',')
    
    return processed_readings


def split_graph(n=16, x_min=-2, x_max=2, y_min=-2, y_max=2):
    """
    This function splits a graph into N equal squares using the provided boundaries.
    It returns a list of tuples, containing the coordinates of the four corners of each square.

    Args:
        n (int, optional):  Defaults to 16.
        x_min (int, optional): Defaults to -2.
        x_max (int, optional):  Defaults to 2.
        y_min (int, optional):  Defaults to -2.
        y_max (int, optional):  Defaults to 2.

    Returns:
        list: list of squares coordinates
    """

    width = (x_max - x_min) / np.sqrt(n)
    height = (y_max - y_min) / np.sqrt(n)

    # Initialize a list to hold the square coordinates
    squares = []

    # Loop through each row and column of squares
    for i in range(int(np.sqrt(n))):
        for j in range(int(np.sqrt(n))):
            # Calculate the x and y coordinates of the square's corners
            x1 = x_min + j * width
            x2 = x_min + (j + 1) * width
            y1 = y_min + i * height
            y2 = y_min + (i + 1) * height

            # Append the square coordinates to the list
            squares.append((x1, x2, y1, y2))

    # Return the list of square coordinates
    return squares


import warnings
warnings.filterwarnings("ignore")

base_dir = "saved_models_bin_structure"
files_models_mapping = {
    "corrected_lidar_1.txt": [
        ("16_squares_results", "run_3-corrected_lidar_1.h5"),
        ("64_squares_results", "run_3-corrected_lidar_1.h5"),
    ],
    "corrected_lidar_1_and_3.txt": [
        ("16_squares_results", "run_3-corrected_lidar_1_and_3.h5"),
        ("64_squares_results", "run_3-corrected_lidar_1_and_3.h5"),
    ],
    "corrected_lidar_1_2_and_3.txt": [
        ("16_squares_results", "run_3-corrected_lidar_1_2_and_3.h5"),
        ("64_squares_results", "run_3-corrected_lidar_1_2_and_3.h5"),
    ],
    "corrected_all_4_lidars.txt": [
        ("16_squares_results", "run_3-corrected_all_4_lidars.h5"),
        ("64_squares_results", "run_3-corrected_all_4_lidars.h5"),
    ],
}

for file, model_paths in files_models_mapping.items():
    for folder, model_file in model_paths:
        if "16" in folder:
            labels = load_results_from_txt("/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/labels.txt")
            results_folder = "16_test_results"
        else:
            labels = load_results_from_txt("/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/labels_64.txt")
            results_folder = "64_test_results"

        os.makedirs(results_folder, exist_ok=True)
        
        squares = split_graph(n=16 if "16" in folder else 64)
        num_quadrants = len(squares)

        print(f"\n\n\n\nReading file {file}...")
        readings = read_readings_from_file(file, "/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/")
        print("Number of readings:", len(readings))

        print("Processing readings...")
        X = process_readings(readings, squares, f"{results_folder}/input_{file}")
        y = np.array(labels)

        # Shuffle readings and labels
        X, y = shuffle_readings(X, y)

        # Convert to numpy arrays and ensure proper shape
        X = np.array(X)
        y = np.array(y)

        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        # Load the trained model
        model_path = os.path.join(base_dir, folder, model_file)
        model = load_model(model_path)

        print(f"### Testing model {model_file} using file {file} with n={16 if '16' in folder else 64} squares... ### ")
        filename = model_file.split('.')[0]
        evaluate_model_with_metrics(model, X, y, filename=os.path.join(results_folder, f"results_{filename}"))
        print(f"### Finished testing model {model_file} using file {file} with n={16 if '16' in folder else 64} squares. ###")