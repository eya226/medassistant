import os
import shutil
import random
from tqdm import tqdm

def create_dataset_split(base_dir, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Scans a directory of class-named subfolders, splits the data, and copies
    it into a new directory with train/val/test subfolders.
    """
    print("--- Starting Data Preparation ---")

    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        print("Please download the dataset from Kaggle and place it in the correct folder.")
        return

    # Ensure a clean start by removing the old processed directory if it exists
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Removing it for a fresh start.")
        shutil.rmtree(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    print(f"Created new directory structure at '{output_dir}'")

    class_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not class_names:
        print(f"Error: No class subdirectories found in '{base_dir}'.")
        return

    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    for class_name in class_names:
        # Create class subdirectories in train, val, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        src_dir = os.path.join(base_dir, class_name)
        image_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"Warning: No images found for class '{class_name}'.")
            continue

        random.shuffle(image_files)

        # Calculate split indices
        train_split = int(len(image_files) * split_ratios[0])
        val_split = int(len(image_files) * (split_ratios[0] + split_ratios[1]))

        # Assign files to sets
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]

        print(f"\nProcessing class: {class_name}")
        print(f"  Total images: {len(image_files)}")
        print(f"  - Training: {len(train_files)}")
        print(f"  - Validation: {len(val_files)}")
        print(f"  - Test: {len(test_files)}")

        # Copy files to new directories
        for file_path in tqdm(train_files, desc=f"  Copying to train/{class_name}"):
            shutil.copy(file_path, os.path.join(train_dir, class_name))

        for file_path in tqdm(val_files, desc=f"  Copying to val/{class_name}"):
            shutil.copy(file_path, os.path.join(val_dir, class_name))

        for file_path in tqdm(test_files, desc=f"  Copying to test/{class_name}"):
            shutil.copy(file_path, os.path.join(test_dir, class_name))

    print("\n--- Data Preparation Complete ---")
    print(f"Data is now organized in '{output_dir}' and ready for model training.")


if __name__ == '__main__':
    # The Kaggle dataset contains 'Training' and 'Testing' folders.
    # We will combine them and then create our own robust split.
    # The user should place the contents of both into a single source folder.

    # Let's define the source and a temporary merge directory
    training_data_path = 'data/Brain Tumor MRI/Training'
    testing_data_path = 'data/Brain Tumor MRI/Testing'
    merged_data_path = 'data/Brain Tumor MRI/merged_source'

    if not os.path.exists(training_data_path) or not os.path.exists(testing_data_path):
        print("Error: The 'Training' or 'Testing' directory from the Kaggle dataset was not found in 'data/Brain Tumor MRI/'.")
        print("Please download and unzip the dataset into the 'data/Brain Tumor MRI/' directory.")
    else:
        # Create a temporary merged directory
        if os.path.exists(merged_data_path):
            shutil.rmtree(merged_data_path)
        os.makedirs(merged_data_path)

        print("Merging 'Training' and 'Testing' directories for a unified split...")
        for folder_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            os.makedirs(os.path.join(merged_data_path, folder_name), exist_ok=True)

            # Copy from training set
            for file in os.listdir(os.path.join(training_data_path, folder_name)):
                shutil.copy(os.path.join(training_data_path, folder_name, file), os.path.join(merged_data_path, folder_name))

            # Copy from testing set
            for file in os.listdir(os.path.join(testing_data_path, folder_name)):
                shutil.copy(os.path.join(testing_data_path, folder_name, file), os.path.join(merged_data_path, folder_name))

        # Run the split on the merged data
        create_dataset_split(base_dir=merged_data_path, output_dir='data/processed_brain_tumor_data')

        # Clean up the temporary merged directory
        print("\nCleaning up temporary merged directory...")
        shutil.rmtree(merged_data_path)
        print("Cleanup complete.")
