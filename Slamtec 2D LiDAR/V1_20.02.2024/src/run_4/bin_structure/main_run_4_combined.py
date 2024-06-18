import os
import numpy as np
import tensorflow
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


def evaluate_model(model, X_test, y_test, filename="results"):
    """
    Epoch: This is the number of the training cycle

    loss: the loss function is a measure of the model's error, or how far the model's predictions are from the true values. Lower values are better.

    accuracy: It is the proportion of correct predictions made out of all predictions. Higher values are better.

    val_loss: This is the value of the loss function for your validation data. Like the training loss, lower values are better. The validation loss gives you an idea of how well your model generalizes to unseen data.

    val_accuracy: This is the accuracy of your model on the validation data. Like the training accuracy, higher values are better.
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

    # Print out the actual and predicted classes for the first ten instances in the test set.
    for i in range(len(y_test)):
        print("Actual square:", y_test[i], "Predicted square:", y_pred[i])

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
    sns.heatmap(general_cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)  # Using a default colormap
    plt.ylabel("Predicted Label", fontsize=14)
    plt.xlabel("Real Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)

    # Define class labels and their positions
    class_labels = ["Positive", "Negative"]
    plt.xticks(ticks=np.arange(2) + 0.5, labels=class_labels, ha="center")
    plt.yticks(ticks=np.arange(2) + 0.5, labels=class_labels, va="center", rotation=0)

    plt.savefig(f"{filename}.png")
    plt.close()


def predict_new_readings(model, readings, squares):
    # Preprocess new readings
    processed_readings = process_readings(readings, squares)

    # Predict probabilities of each class
    pred_prob = model.predict(processed_readings)

    # Convert predicted probabilities into actual class predictions by setting a threshold
    pred = (pred_prob > 0.5).astype(int)

    return pred


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


# Function to save the trained model to a folder
def save_model(model, model_name, model_dir="saved_models"):
    """
    Saves the given model under the specified directory with the model name.

    :param model: Trained Keras model to be saved.
    :param model_name: Name of the model to be used as the filename.
    :param model_dir: Directory where the model will be saved.
    """
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")


# Function to load a model from a folder
def load_trained_model(model_path):
    """
    Loads a Keras model from the specified path.

    :param model_path: Path to the model .h5 file to be loaded.
    :return: Loaded Keras model.
    """
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def plot_2d_graph_with_squares(readings, squares):
    """
    Plots a scatter plot of multiple points and overlays rectangles (squares) on the plot.

    Args:
      readings: A list of tuples where each tuple contains the x and y coordinates.
      squares: List of tuples representing squares. Each tuple contains four values:
               the x-coordinate of the left side, the x-coordinate of the right side,
               the y-coordinate of the bottom side, and the y-coordinate of the top side of the square.
    """
    fig, ax = plt.subplots()

    # Plot all points in the readings
    x_vals, y_vals = zip(*readings)
    ax.scatter(x_vals, y_vals, s=1, c="black")

    # Overlay squares
    for i, square in enumerate(squares):
        rect = plt.Rectangle(
            (square[0], square[2]),
            square[1] - square[0],
            square[3] - square[2],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            (square[0] + square[1]) / 2,
            (square[2] + square[3]) / 2,
            str(i),
            fontsize=10,
            color="r",
            ha="center",
            va="center",
        )

    # Set the x and y limits of the plot
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Set the aspect ratio to equal
    ax.set_aspect("equal")

    # Set the x and y labels and title of the plot
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("LIDAR Mapping")

    plt.show()


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


runs = 16000
# List of files to iterate through for training
files = [
    "corrected_lidar_1.txt",
    "corrected_lidar_1_and_3.txt",
    "corrected_lidar_1_2_and_3.txt",
    "corrected_all_4_lidars.txt",
]

# Loop for different square sizes
for n in [16, 64]:
    # Load corresponding labels
    if n == 16:
        labels = load_results_from_txt("/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/labels.txt")
        results_folder = "16_squares_results"
    else:
        labels = load_results_from_txt("/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/labels_64.txt")
        results_folder = "64_squares_results"

    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)
    
    # Split graph into squares
    squares = split_graph(n=n)
    num_quadrants = len(squares)

    for file in files:
        # ----------------------------- Getting readings ----------------------------- #
        # Get array of readings
        print(f"\n\n\n\nReading file {file}...")
        readings = read_readings_from_file(file, "/home/lidar/Documents/Matheus/Slamtec 2D LiDAR/codes/data/run_2/")

        print("Number of readings:", len(readings))

        # Process your readings
        print("Processing readings...")
        X = process_readings(readings, squares, f"{results_folder}/input_{n}_{file}")  # Input
        y = np.array(labels)  # Labels
        
        # Shuffle readings and labels
        X, y = shuffle_readings(X, y)

        # Convert to numpy arrays and ensure proper shape
        X = np.array(X)
        y = np.array(y)

        # Split your data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define your model
        model = Sequential()
        model.add(Dense(128, activation="relu", input_shape=(len(squares),)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_quadrants, activation="sigmoid"))

        # Compile your model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train your model
        print(f"Training model for {file} with {n} squares...")
        model.fit(X_train, y_train, epochs=runs, validation_data=(X_test, y_test))

        # ---------------------- Evaluation and result checking ---------------------- #
        filename = file.split('.')[0]  # Use the file name as part of the results file name
        evaluate_model(model, X_test, y_test, filename=os.path.join(results_folder, f"results_{filename}"))

        # After training your model
        save_model(model, f"{results_folder}/run_4-{filename}")


