# Deep Learning Medical Image Classifier

## Overview

This project is a full-stack deep learning application designed to analyze chest X-ray images for the detection of pneumonia. It uses a powerful pre-trained Convolutional Neural Network (CNN) to classify images and provides an intuitive web interface for users to get instant, interpretable results.

## Features

*   **Pneumonia Classification:** Classifies chest X-rays as either "Normal" or "Pneumonia" with a high degree of accuracy.
*   **Visual Heatmap Analysis (Grad-CAM):** For "Pneumonia" predictions, the application generates a visual heatmap overlay on the X-ray. This highlights the specific regions in the image that most influenced the model's decision, providing valuable insight into the AI's reasoning.
*   **Written Analysis Summary:** Accompanies each prediction with a clear, text-based summary of the findings, including the prediction, confidence score, and a brief interpretation.
*   **Interactive Web Interface:** A clean, user-friendly web application that allows for easy image uploads and clear presentation of the results.

## Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, OpenCV, Scikit-learn
*   **Frontend:** HTML, CSS, JavaScript
*   **Testing:** Pytest

---

## Setup and Launch Instructions

Follow these steps carefully to set up and run the project on your local machine.

### 1. Prerequisites

*   **Python 3.11:** This project is developed and tested with Python 3.11. Using other versions (especially newer ones like 3.12+) may cause issues with TensorFlow compatibility. You can download Python 3.11 [here](https://www.python.org/downloads/release/python-3118/).
*   **Git:** For cloning the repository.
*   **(For Windows Users) Microsoft C++ Build Tools:** You may need to install the C++ build tools if you encounter errors during dependency installation. You can get them from the [Visual Studio website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### 2. Clone the Repository

Open your terminal or command prompt and run the following command:
```bash
git clone <repository_url>
cd <repository_folder>
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Once your virtual environment is active, install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

This project uses the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle. You must download it manually.

1.  **Go to the dataset page:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2.  **Download** the dataset (a Kaggle account is required).
3.  **Unzip** the downloaded file (`archive.zip`).
4.  Place the `chest_xray` directory inside the `data/` directory in the project root. The final folder structure should be:
    ```
    .
    ├── data/
    │   └── chest_xray/
    └── ... (other project files)
    ```

### 6. Run Data Preparation

This script processes the raw images into the correct format for the model.

```bash
python prepare_data.py
```

### 7. Train the Model

This script trains the model using the prepared data. The best-performing model will be saved in the `saved_model/` directory.

```bash
python train_model.py
```
*(Note: This step is computationally intensive and may take a significant amount of time.)*

### 8. Launch the Application

Once the model is trained, you can launch the web application.

```bash
python app/main.py
```

The server will start. Open your web browser and go to the following address: **http://127.0.0.1:5000**

## How to Use the Application

1.  Click the "Analyze X-Ray Image" button.
2.  Drag and drop an X-ray image file onto the designated area, or click to open the file explorer.
3.  The application will process the image and display the results, including the prediction, confidence score, a written analysis, and a visual heatmap if pneumonia is detected.
