import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration ---
DATA_DIR = 'data/processed_brain_tumor_data'
MODEL_OUTPUT_DIR = 'saved_model'
IMG_SIZE = (150, 150) # Using a slightly larger image size for more detail
BATCH_SIZE = 16 # Reduced batch size to prevent memory errors
NUM_CLASSES = 4 # glioma, meningioma, notumor, pituitary
EPOCHS = 30

# --- Data Generators ---
def create_data_generators(base_dir):
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found. Please run prepare_data.py first.")
        return None, None, None

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

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
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
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
def build_brain_tumor_model(num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Main Execution ---
if __name__ == '__main__':
    generators = create_data_generators(DATA_DIR)
    if generators:
        train_gen, val_gen, test_gen = generators

        model = build_brain_tumor_model(NUM_CLASSES)
        model.summary()

        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_OUTPUT_DIR, 'brain_tumor_model.keras'),
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

        print("\n--- Starting Brain Tumor Model Training ---")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[checkpoint, early_stopping]
        )

        class_indices = {v: k for k, v in train_gen.class_indices.items()}
        with open(os.path.join(MODEL_OUTPUT_DIR, 'brain_tumor_class_indices.json'), 'w') as f:
            json.dump(class_indices, f)
        print(f"\nClass indices saved to 'brain_tumor_class_indices.json'")

        print("\n--- Evaluating Final Model on Test Set ---")
        test_loss, test_acc = model.evaluate(test_gen, verbose=2)
        print(f'\nTest Accuracy: {test_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
