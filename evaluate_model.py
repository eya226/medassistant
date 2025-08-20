import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf

# Define constants
DATA_DIR = 'data'
MODEL_PATH = 'saved_model/best_model.keras'
RESULTS_DIR = 'evaluation_results'

def load_test_data():
    """Loads the preprocessed test data."""
    try:
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        return X_test, y_test
    except FileNotFoundError:
        print("Error: Test data not found.")
        print("Please run 'prepare_data.py' first to generate the data.")
        return None, None

def plot_confusion_matrix(y_true, y_pred_classes, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_pred_probs, save_path):
    """Plots and saves the AUC-ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Load data and model
    X_test, y_test = load_test_data()

    if X_test is not None and os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Make predictions
        print("Making predictions on the test set...")
        y_pred_probs = model.predict(X_test).ravel()
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        # --- Evaluation ---

        # 1. Classification Report (Precision, Recall, F1-score)
        print("\n--- Classification Report ---")
        report = classification_report(y_test, y_pred_classes, target_names=['NORMAL', 'PNEUMONIA'])
        print(report)
        # Save report to a text file
        with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        print(f"Classification report saved to {os.path.join(RESULTS_DIR, 'classification_report.txt')}")

        # 2. Confusion Matrix
        print("\n--- Generating Confusion Matrix ---")
        cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred_classes, cm_path)

        # 3. AUC-ROC Curve
        print("\n--- Generating AUC-ROC Curve ---")
        roc_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
        plot_roc_curve(y_test, y_pred_probs, roc_path)

        print("\nModel evaluation complete.")

    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        print("Please run 'train_model.py' first to train and save the model.")
    else:
        # Error message from load_test_data() is already printed
        pass
