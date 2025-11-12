import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import math 

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-5 
EPOCHS = 30       
BASE_DATA_DIR = './dataset/' # Assumes ./dataset/train and ./dataset/validation
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VALID_DIR = os.path.join(BASE_DATA_DIR, 'validation')


def build_baseline_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    """
    Builds the ResNet-50 baseline model for transfer learning.
    
    The base model layers are frozen, and a new classification
    head is added.
    """
 
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape)

    base_model.trainable = False

    # 3. Add the new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Pool features
    x = Dense(128, activation='relu')(x) # Add a dense hidden layer
    x = Dropout(0.5)(x) # Add dropout for regularization
    
    # 4. Add the final output layer
    # Sigmoid activation for binary (0 or 1) classification
    predictions = Dense(1, activation='sigmoid')(x)

    # 5. Create the new model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def get_data_generators():
    """
    Sets up the training and validation data generators
    with data augmentation for the training set.
    """
    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalize pixel values
        rotation_range=10,         # Random rotation
        width_shift_range=0.1,     # Random horizontal shift
        height_shift_range=0.1,    # Random vertical shift
        shear_range=0.1,           # Shear
        zoom_range=0.1,            # Random zoom
        horizontal_flip=True,      # Horizontal flip (check if medically appropriate)
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary' 
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False 
    )
    
    return train_generator, validation_generator

def main():
    model = build_baseline_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()

    try:
        train_gen, valid_gen = get_data_generators()
        
        label_0 = train_gen.class_indices['Normal'] 
        label_1 = train_gen.class_indices['Scol'] 

        total_samples = train_gen.samples
        count_0 = np.sum(train_gen.labels == label_0)
        count_1 = np.sum(train_gen.labels == label_1)
        
        weight_0 = (1 / count_0) * (total_samples / 2.0)
        weight_1 = (1 / count_1) * (total_samples / 2.0)

        class_weights = {label_0: weight_0, label_1: weight_1}
        
        print("\n--- CLASS WEIGHTS ---")
        print(f"Total Training Samples: {total_samples}")
        print(f"Normal (Class 0) Count: {count_0} -> Weight: {weight_0:.2f}")
        print(f"Scol (Class 1) Count: {count_1} -> Weight: {weight_1:.2f}")
        print("-----------------------\n")
        # ------------------------------------------------------------------

        # Train the model
        print("\n--- Starting Initial Model Training (Frozen Layers) ---")
        
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS,
            class_weight=class_weights 
        )
        
        print("\n--- Initial Training Complete ---")
        
     
        model.save('scoliosis_baseline_model.h5')
        print("Model saved to 'scoliosis_baseline_model.h5'")

    except FileNotFoundError:
        print(f"Error: Data directories not found.")
        print(f"Please ensure your data is structured in:")
        print(f"{TRAIN_DIR}")
        print(f"{VALID_DIR}")

if __name__ == '__main__':
    main()
