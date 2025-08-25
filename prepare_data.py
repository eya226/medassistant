import os
import shutil
import random
from tqdm import tqdm

def create_train_val_test_split(base_dir='data/Chest Diseases Data',
                                output_dir='data/processed_data',
                                split_ratios=(0.8, 0.1, 0.1)):
    """
    Scans a directory of class-named subfolders, splits the data, and copies
    it into a new directory with train/val/test subfolders.
    """
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        print("Please download the dataset and place it in the 'data/' folder.")
        return

    # Create the output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Deleting it to ensure a fresh start.")
        shutil.rmtree(output_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

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

        # Get all image file paths for the current class
        src_dir = os.path.join(base_dir, class_name)
        # The dataset has an extra layer of subdirectories, e.g., Atelectasis/Atelectasis
        # We need to handle this. Let's assume the images are in the deepest directory.
        image_files = []
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    image_files.append(os.path.join(root, file))

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

        # Copy files to new directories
        print(f"\nProcessing class: {class_name}")
        print(f"  Total images: {len(image_files)}")
        print(f"  Training: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

        for file_path in tqdm(train_files, desc=f"  Copying to train/{class_name}"):
            shutil.copy(file_path, os.path.join(train_dir, class_name))

        for file_path in tqdm(val_files, desc=f"  Copying to val/{class_name}"):
            shutil.copy(file_path, os.path.join(val_dir, class_name))

        for file_path in tqdm(test_files, desc=f"  Copying to test/{class_name}"):
            shutil.copy(file_path, os.path.join(test_dir, class_name))

    print("\nData splitting and copying complete.")
    print(f"Your data is now organized in '{output_dir}' and ready for training.")


if __name__ == '__main__':
    # The expected name of the dataset folder after unzipping from Kaggle
    dataset_base_dir = 'data/Chest Diseases Data'
    create_train_val_test_split(base_dir=dataset_base_dir)
