import os
from PIL import Image
import numpy as np
import shutil
import pandas as pd
import glob

REQUIRED_WIDTH = 200
REQUIRED_HEIGHT = 250
IMAGE_DIRECTORY = "/Users/darshabi/Desktop/TEST/augmented_folder"
TRAIN_DIRECTORY = "/Users/darshabi/Desktop/TEST/augmented_folder/train"
TEST_DIRECTORY = "/Users/darshabi/Desktop/TEST/augmented_folder/test"
OUTPUT_TRAIN_PICKLE = "/Users/darshabi/Desktop/ITC/Final_Project/train"
OUTPUT_TEST_PICKLE = "/Users/darshabi/Desktop/ITC/Final_Project/test"


def split_images(directory, train_directory, test_directory, split_ratio=0.7):
    """
    Split the images in the given directory into train and test directories.

    Args:
        directory (str): Path to the directory containing the images.
        train_directory (str): Path to the train directory to create.
        test_directory (str): Path to the test directory to create.
        split_ratio (float): Ratio of images to be used for training.

    Returns:
        tuple: Paths of the train and test directories.
    """
    # Create train and test directories if they don't exist
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    # Get the list of image files in the directory
    image_files = [filename for filename in os.listdir(directory)
                   if filename.endswith(".jpg") or filename.endswith(".png")]

    # Calculate the number of images for training and testing
    num_images = len(image_files)
    num_train = int(num_images * split_ratio)

    # Shuffle the image files randomly
    np.random.shuffle(image_files)

    # Move the images to the train and test directories
    for i, filename in enumerate(image_files):
        file_path = os.path.join(directory, filename)
        if i < num_train:
            shutil.move(file_path, os.path.join(train_directory, filename))
        else:
            shutil.move(file_path, os.path.join(test_directory, filename))

    # Return the paths of the train and test directories
    return train_directory, test_directory


def preprocess_images(directory, output_pickle, batch_size=1000):
    """
    Preprocess the images in the given directory and save them in batches as pickles.

    Args:
        directory (str): Path to the directory containing the images.
        output_pickle (str): Path and prefix for the output pickle files.
        batch_size (int): Number of images to be saved per batch (default: 1000).

    """
    labels = []
    images = []

    for i, filename in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        try:
            label = int(filename[0])

            with Image.open(file_path) as image:
                width, height = image.size

                if width > height:
                    image = image.rotate(angle=90)

                image = image.resize((REQUIRED_WIDTH, REQUIRED_HEIGHT))
                pix = np.array(image)

                labels.append(label)
                images.append(pix.flatten())

                print(f"Image {filename} preprocessed.")

        except IOError:
            print(f"Failed to open or process image: {filename}")

        # If batch size is reached, save and clear lists
        if (i+1) % batch_size == 0:
            batch_num = (i+1) // batch_size
            np.savez(f"{output_pickle}_batch_{batch_num}", labels=np.array(labels), images=np.array(images))
            print(f"Preprocessed data saved to {output_pickle}_batch_{batch_num}.npz")
            labels.clear()
            images.clear()

    # Save any remaining data
    if labels and images:
        batch_num = len(os.listdir(directory)) // batch_size + 1
        np.savez(f"{output_pickle}_batch_{batch_num}", labels=np.array(labels), images=np.array(images))
        print(f"Preprocessed data saved to {output_pickle}_batch_{batch_num}.npz")

    print(f"All data preprocessed and saved in batches.")


def concatenate_npz_files(file_prefix, output_file):
    """
    Concatenate the NPZ files with the given prefix into a single dataframe and save it.

    Args:
        file_prefix (str): Prefix of the NPZ files to be concatenated.
        output_file (str): Path and filename for the output file (CSV or pickle).

    """
    dataframes = []

    # Look for files that start with the given prefix and end with .npz
    npz_files = glob.glob(f"{file_prefix}_batch_*.npz")

    for npz_file in sorted(npz_files):
        with np.load(npz_file) as data:
            df = pd.DataFrame(data["images"])
            df.insert(0, "label", data["labels"])
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # If output file is csv
    if output_file.endswith('.csv'):
        combined_df.to_csv(output_file, index=False)
    # If output file is pickle
    elif output_file.endswith('.pkl'):
        combined_df.to_pickle(output_file)

    print(f"All data saved to {output_file}.")


if __name__ == "__main__":
    train_dir, test_dir = split_images(IMAGE_DIRECTORY, TRAIN_DIRECTORY, TEST_DIRECTORY)
    preprocess_images(train_dir, OUTPUT_TRAIN_PICKLE, batch_size=1000)
    preprocess_images(test_dir, OUTPUT_TEST_PICKLE, batch_size=1000)
    concatenate_npz_files(OUTPUT_TRAIN_PICKLE, "final_train_data.pkl")
    concatenate_npz_files(OUTPUT_TEST_PICKLE, "final_test_data.pkl")
