import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns


def missing_values_percentage(df):
    """
    Calculate the percentage of missing values in each column and print the results.

    Parameters:
        df (DataFrame): The DataFrame containing the data.

    """
    # Calculate the percentage of missing values in each column
    missing_percentages = df.isnull().sum() / len(df) * 100

    # Print the percentage of missing values for each column
    for column, percentage in missing_percentages.iteritems():
        print(f"\nColumn '{column}' has {percentage:.2f}% missing values.")


def basic_eda(df):
    """
    Perform basic exploratory data analysis (EDA) on the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.

    """
    print(df.head())
    # Check the dimensions of the dataset (number of rows and columns)
    print("\nDataset dimensions:", df.shape)

    # Get a summary of the dataset
    print(df.info())

    # Get basic statistics of the dataset
    print(df.describe())

    if df.isnull().sum().sum() != 0:
        missing_values_percentage(df)
    else:
        print("No Missing Values Found.")


def plot_label_distribution(df):
    """
    Plot the distribution of labels in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
    """
    # Calculate the count of each unique label
    label_counts = df['label'].value_counts()

    # Create a bar plot with index as x and values as height
    label_counts.plot(kind='bar')

    # Set title and labels
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()


def plot_sample_images(df, REQUIRED_WIDTH=200, REQUIRED_HEIGHT=250):
    """
    Plot one sample image from each class.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
    """
    labels = df['label'].unique()
    label_descriptions = {
        0: "Scooter is parked properly",
        1: "Picture does not contain scooter",
        2: "Picture is not clear enough due to blur",
        3: "Picture is not clear enough due distance from scooter",
        4: "Scooter is not parked upright (is on it's side)",
        5: "Scooter blocks footpath"
    }

    for label in labels:
        sample_image = df[df['label'] == label].sample(n=1)

        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        for _, row in sample_image.iterrows():
            image_data = np.array(row[1:], dtype='uint8').reshape((REQUIRED_HEIGHT, REQUIRED_WIDTH, 3))
            axes.imshow(image_data)
            axes.axis('off')
            axes.set_title(f'Label {label}: {label_descriptions[label]}')

        plt.show()


def plot_color_histograms(df, REQUIRED_WIDTH=200, REQUIRED_HEIGHT=250):
    """
    Plot histograms showing the distribution of different colors (RGB channels) in the images for each class.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        REQUIRED_WIDTH (int): Required width of the images.
        REQUIRED_HEIGHT (int): Required height of the images.
    """
    labels = df['label'].unique()

    for label in labels:
        sample_images = df[df['label'] == label].sample(n=3)

        fig, axes = plt.subplots(3, 1, figsize=(8, 8))

        for i, (_, row) in enumerate(sample_images.iterrows()):
            image_data = np.array(row[1:], dtype='uint8').reshape((REQUIRED_HEIGHT, REQUIRED_WIDTH, 3))

            # Calculate RGB histograms
            hist_r, bins_r = np.histogram(image_data[:, :, 0].flatten(), bins=256, range=[0, 256])
            hist_g, bins_g = np.histogram(image_data[:, :, 1].flatten(), bins=256, range=[0, 256])
            hist_b, bins_b = np.histogram(image_data[:, :, 2].flatten(), bins=256, range=[0, 256])

            # Plot RGB histograms
            axes[i].plot(bins_r[:-1], hist_r, color='red', alpha=0.7)
            axes[i].plot(bins_g[:-1], hist_g, color='green', alpha=0.7)
            axes[i].plot(bins_b[:-1], hist_b, color='blue', alpha=0.7)
            axes[i].set_xlim([0, 256])
            axes[i].set_ylim([0, max(max(hist_r), max(hist_g), max(hist_b))])
            axes[i].set_title(f'Label {label}, Image {i+1} Color Histograms')
            axes[i].set_xlabel('Pixel Intensity')
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


def main():
    with open('final_train_data.pkl', 'rb') as file:
        data = pickle.load(file)
    train = pd.DataFrame(data)

    basic_eda(train)
    plot_label_distribution(train)
    plot_sample_images(train)
    plot_color_histograms(train)


if __name__ == "__main__":
    main()
