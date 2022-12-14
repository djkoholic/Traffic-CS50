import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line argument
    if len(sys.argv) not in [2, 3, 4]:
        sys.exit("Usage: python traffic.py data_directory test_image [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Predict test image

    if len(sys.argv) == 3:
        print(f"Given image belongs to {predict(model)} class.")

    # Save model to file
    if len(sys.argv) == 4:
        filename = sys.argv[3]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    current_dir = os.getcwd()
    absolute_dir = os.path.join(current_dir,data_dir)
    
    new_dim = (IMG_WIDTH, IMG_HEIGHT)

    total_no_of_files = 0
    for i in range(NUM_CATEGORIES):
        category_path = os.path.join(absolute_dir, str(i))
        files = os.listdir(category_path)
        total_no_of_files += len(files)

    images = [np.zeros((IMG_WIDTH, IMG_HEIGHT, 3)) for i in range(total_no_of_files)]
    labels = [-1 for i in range(total_no_of_files)]
    file_count = 0
    for i in range(NUM_CATEGORIES):
        category_path = os.path.join(absolute_dir, str(i))
        files = os.listdir(category_path)
        for file in files:
            file_path = os.path.join(category_path, file)
            orig_img = cv2.imread(file_path)
            resized_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_AREA)
            images[file_count] = resized_img
            labels[file_count] = i
            file_count += 1
    return(images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3,3), activation="relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def predict(model):
    img_test = cv2.imread(sys.argv[2])
    img_test2 = cv2.resize(img_test,(IMG_WIDTH, IMG_HEIGHT),interpolation=cv2.INTER_AREA)
    img_test3 = np.array(img_test2)
    classification = model.predict(
            [np.reshape(img_test3, (1, IMG_HEIGHT, IMG_WIDTH, 3))]
        ).argmax()
    return classification

if __name__ == "__main__":
    main()
