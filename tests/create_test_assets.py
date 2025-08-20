import numpy as np
import cv2
import os

def create_dummy_image(path='tests/dummy_image.png', size=(128, 128)):
    """Creates a simple grayscale dummy image."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # Create a black image
    image = np.zeros((size[1], size[0]), dtype=np.uint8)
    # Add some white noise
    cv2.randu(image, 0, 255)
    # Write the image to a file
    cv2.imwrite(path, image)
    print(f"Dummy image created at {path}")

def create_dummy_text_file(path='tests/dummy_file.txt'):
    """Creates a simple text file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write("This is not an image.")
    print(f"Dummy text file created at {path}")


if __name__ == '__main__':
    create_dummy_image()
    create_dummy_text_file()
