import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration ---
DATA_DIR = 'data/processed_data'
MODEL_OUTPUT_DIR = 'saved_model'
IMG_SIZE = (128, 128) # Use a tuple for image size
BATCH_SIZE = 32
NUM_CLASSES = 6 # We have 6 classes
EPOCHS = 30 # Increased epochs for this more complex task

# --- Data Generators ---
def create_data_generators(base_dir):
    """Creates data generators for training, validation, and testing."""
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical' # for multi-class classification
    )

    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

# --- Model Building ---
def build_multi_class_model(num_classes):
    """Builds a multi-class model using DenseNet121 for transfer learning."""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom top layers for multi-class classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    # The final layer has 'num_classes' neurons and 'softmax' activation
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', # Loss function for multi-class
        metrics=['accuracy']
    )

    return model

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        print(f"Error: Processed data directory not found at '{DATA_DIR}'")
        print("Please run 'prepare_data.py' first to create the train/val/test splits.")
    else:
        train_gen, val_gen, test_gen = create_data_generators(DATA_DIR)

        model = build_multi_class_model(NUM_CLASSES)
        model.summary()

        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_OUTPUT_DIR, 'best_multiclass_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        print("\nStarting model training...")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[checkpoint, early_stopping]
        )

        print("\nModel training complete.")

        # Save the class indices for use in the application
        import json
        class_indices = train_gen.class_indices
        # Invert the dictionary to map from index to class name
        inv_class_indices = {v: k for k, v in class_indices.items()}
        with open(os.path.join(MODEL_OUTPUT_DIR, 'class_indices.json'), 'w') as f:
            json.dump(inv_class_indices, f)
        print(f"\nClass indices saved to '{os.path.join(MODEL_OUTPUT_DIR, 'class_indices.json')}'")

        print("\nEvaluating model on the test set...")
        test_loss, test_acc = model.evaluate(test_gen, verbose=2)
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}')
