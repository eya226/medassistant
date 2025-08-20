import os
import cv2
import numpy as np
from tqdm import tqdm

# Define constants
IMG_SIZE = 128
DATA_DIR = 'data/chest_xray'
OUTPUT_DIR = 'data'

# Define paths for train, validation, and test sets
train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'test')

def process_data(data_dir, desc):
    """
    Loads, resizes, and normalizes images from a directory.
    Args:
        data_dir (str): The path to the directory (e.g., train, val, test).
        desc (str): Description for the progress bar.
    Returns:
        A tuple of (images, labels).
    """
    images = []
    labels = []

    normal_dir = os.path.join(data_dir, 'NORMAL')
    pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')

    for img_file in tqdm(os.listdir(normal_dir), desc=f'Processing NORMAL in {desc}'):
        img_path = os.path.join(normal_dir, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append(0) # 0 for NORMAL
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    for img_file in tqdm(os.listdir(pneumonia_dir), desc=f'Processing PNEUMONIA in {desc}'):
        img_path = os.path.join(pneumonia_dir, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append(1) # 1 for PNEUMONIA
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return np.array(images), np.array(labels)

if __name__ == '__main__':
    # Check if the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please download the dataset from Kaggle and place it in the 'data/' directory as instructed.")
    else:
        print("Processing training data...")
        X_train, y_train = process_data(train_dir, 'train')

        print("\nProcessing validation data...")
        X_val, y_val = process_data(val_dir, 'val')

        print("\nProcessing test data...")
        X_test, y_test = process_data(test_dir, 'test')

        # Reshape data for the model (add channel dimension)
        X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        # Save the processed data
        print(f"\nSaving processed data to {OUTPUT_DIR}...")
        np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
        np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

        print("\nData preparation complete.")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
