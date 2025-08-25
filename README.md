# AI Medical Triage System

## Overview

This project is a full-stack deep learning application designed to function as an intelligent triage system for lung diseases. It analyzes chest CT scans to detect and classify multiple conditions, then presents the results in a prioritized list to help medical professionals focus on the most urgent cases first.

## Features

*   **Multi-Disease Classification:** Classifies chest CT scans into 6 categories: Atelectasis, Edema, Pneumonia, Pneumothorax, Tuberculosis, and Normal.
*   **Triage-Based Sorting:** The application accepts multiple scans at once and sorts the results based on a predefined clinical urgency, placing the most severe conditions at the top of the list.
*   **Visual Heatmap Analysis (Grad-CAM):** For positive disease predictions, the application generates a visual heatmap on the CT scan. This highlights the specific regions that most influenced the model's decision, providing valuable interpretability.
*   **Written Analysis Summary:** Accompanies each prediction with a clear, text-based summary of the findings.
*   **Interactive Web Interface:** A clean, user-friendly web application that allows for easy batch uploading of images and clear presentation of the sorted triage list.

## Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, OpenCV
*   **Frontend:** HTML, CSS, JavaScript
*   **Testing:** Pytest

---

## Setup and Launch Instructions

Follow these steps carefully to set up and run the project on your local machine.

### 1. Prerequisites

*   **Python 3.11:** This project is developed and tested with Python 3.11. Using other versions may cause issues with TensorFlow compatibility.
*   **Git:** For cloning the repository.

### 2. Clone the Repository

Open your terminal and run the standard `git clone` command for this repository.

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies. From the project's root directory:

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

Once your virtual environment is active, install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

This project uses the "MultiClass Pulmonary Disease CT Image Dataset" from Kaggle.

1.  **Go to the dataset page:** [https://www.kaggle.com/datasets/programmer3/chest-diseases-by-medical-imaging](https://www.kaggle.com/datasets/programmer3/chest-diseases-by-medical-imaging)
2.  **Download** the dataset.
3.  **Unzip** the downloaded file. The folder will be named `Chest Diseases Data`.
4.  Place this `Chest Diseases Data` directory inside the `data/` directory in the project root.

### 6. Prepare the Data

This script organizes the dataset into `train`, `val`, and `test` folders.

```bash
python prepare_data.py
```

### 7. Train the Model

This script trains the multi-class model on the prepared data. The best model will be saved in the `saved_model/` directory.

```bash
python train_model.py
```
*(Note: This step is computationally intensive and may take a significant amount of time.)*

### 8. Launch the Application

Once the model is trained, launch the web application.

```bash
python app/main.py
```

The server will start. Open your web browser and go to: **http://127.0.0.1:5000**
