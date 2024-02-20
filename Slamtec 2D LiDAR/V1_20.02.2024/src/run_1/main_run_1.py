import numpy as np
import tensorflow
from utils import GraphHandler as gh
from utils import CoordinatesHandler as ch
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------- Auxiliary functions --------------------------- #
def process_readings(readings, squares):
    """
    The function takes a list of readings in polar coordinates and a list of squares, converts the
    readings to Cartesian coordinates, and determines which square each point falls into.
    
    :param readings: The readings parameter is a list of tuples. Each tuple contains two lists: thetas
    and dists. thetas represents the angles at which the readings were taken, and dists represents the
    corresponding distances from the origin
    :param squares: The "squares" parameter is a list of tuples, where each tuple represents the
    boundaries of a square in Cartesian coordinates. Each tuple contains four values: x1, x2, y1, y2.
    These values represent the minimum and maximum x and y coordinates of the square, respectively
    :return: a 2D numpy array called `processed_readings`.
    """
    # Convert the readings to Cartesian coordinates and determine the square for each point
    # Initialize an empty array for the processed readings
    processed_readings = np.zeros((len(readings), len(squares)))

    for i, (thetas, dists) in enumerate(readings):
        for theta, dist in zip(thetas, dists):
            x, y = ch.polar_to_cartesian(theta, dist)

            # Determine which square this point is in
            for j, (x1, x2, y1, y2) in enumerate(squares):
                if x1 <= x < x2 and y1 <= y < y2:
                    processed_readings[i][j] = 1
                    break

    return processed_readings

def convert_quadrants_to_binary(quadrants, num_quadrants):
    num_readings = len(quadrants)

    # Initialize a new array for the target data
    y = np.zeros((num_readings, num_quadrants), dtype=int)

    # Set the corresponding elements to 1 for each reading
    for i in range(num_readings):
        if quadrants[i][0] != -1:  # if people are detected in the reading
            for quadrant in quadrants[i]:
                y[i, quadrant] = 1
    return y

def evaluate_model(model, X_test, y_test):
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

    # Calculate the precision of the model's predictions. Precision is the ratio of correctly predicted positive observations to the total predicted positives.
    precision = precision_score(
        y_test, y_pred, average='samples', zero_division=1)

    # Calculate the recall (sensitivity) of the model's predictions. Recall is the ratio of correctly predicted positive observations to the all observations in actual class.
    recall = recall_score(y_test, y_pred, average='samples', zero_division=1)

    # Calculate the F1 score of the model's predictions. The F1 Score is the weighted average of Precision and Recall.
    f1 = f1_score(y_test, y_pred, average='samples', zero_division=1)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def predict_new_readings(model, readings, squares):
    # Preprocess new readings
    processed_readings = process_readings(readings, squares)
    
    # Predict probabilities of each class
    pred_prob = model.predict(processed_readings)
    
    # Convert predicted probabilities into actual class predictions by setting a threshold
    pred = (pred_prob > 0.5).astype(int)
    
    return pred

# ------------------------- Getting quadrants answers ------------------------ #
p_quadrants_1 = np.array([
    [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3], [13, 3]], dtype=object)

p_quadrants_2 = np.array([
    [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [1, 11], [1, 11], [1, 11], [0, 7], [0, 11], [0, 11], [0, 7], [1, 7], [1, 7], [1, 7], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1, 11], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 7], [1, 11], [1, 11], [1, 11], [1, 11], [1, 11], [1, 7], [1, 7], [1, 7]], dtype=object)

# Concatenate the arrays
quadrants = np.concatenate((p_quadrants_1, p_quadrants_2))

# ----------------------- Dividing x and y into squares ---------------------- #
# Fixed square size of 1x1m, that is, 16 squares (Default values)
squares = ch.split_graph()
num_quadrants = len(squares)

# ----------------------------- Getting readings ----------------------------- #
commonLocation = "C:\\Users\\mathe\\Documents\\Projects\\IC-6G\\project\\slamtec\\samples\\run_1\\"
sample1 = commonLocation + "p_reading_1.txt"
sample2 = commonLocation + "p_reading_2.txt"
sample3 = commonLocation + "p_reading_3.txt"
sample4 = commonLocation + "random_reading_1.txt"
sample5 = commonLocation + "random_reading_2.txt"
sample6 = commonLocation + "random_reading_3.txt"
sample7 = commonLocation + "random_reading_4.txt"

# Get array of readings
print('Reading file...')
readings = ch.read_multiple_samples_slamtec(
    sample1) + ch.read_multiple_samples_slamtec(sample2)
print('Number of readings:', len(readings))

# Filtering readings by setting maximum range
readings = ch.filter_readings(readings, -2, 2, -2, 2)  # square 4x4

# ----------------------------- Using tensorFlow ----------------------------- #
runs = 1  # Number os training cycles

# Define your model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(squares),)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_quadrants, activation='sigmoid'))

# Compile your model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Process your readings
print('Processing readings...')
X = process_readings(readings, squares)  # Input
y = convert_quadrants_to_binary(quadrants, num_quadrants)  # Labels
# [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train your model
print('Training model...')
model.fit(X_train, y_train, epochs=runs, validation_data=(X_test, y_test))

# ---------------------- Evaluation and result checking ---------------------- #
evaluate_model(model, X_test, y_test)
print("\n# ------------------------------ Testing it out ------------------------------ #\n")

# ------------------------------ Testing it out ------------------------------ #
# Get array of readings
print('Reading new file...')
readings = ch.read_multiple_samples_slamtec(sample1)
print('Number of readings:', len(readings))

# Filtering readings by setting maximum range
readings = ch.filter_readings(readings, -2, 2, -2, 2)  # square 4x4

# Predict using the trained model
print('Using trained model...')
new_predictions = predict_new_readings(model, readings, squares)
for i in range(len(new_predictions)):
    print("Predicted square:", new_predictions[i])
    
    # ------------ reading each theta and r, then transforming to x, y ----------- #
    theta, r = readings[i]
    x, y = ch.polar_to_cartesian(r, theta)
    
    # --------------------------------- Plotting --------------------------------- #
    gh.plot_2d_graph_with_squares(x, y, squares, window_size=9) 
    
    
    
from keras.models import load_model
import os

# Function to save the trained model to a folder
def save_model(model, model_name, model_dir='saved_models'):
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

# After training your model
save_model(model, "my_model")

# When you need to use the model
loaded_model = load_trained_model("saved_models/my_model.h5")