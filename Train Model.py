import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt

# Path to the dataset
dataset_path = '/content/preprocessing_dataset'  # Adjust this path as needed

# Parameters
img_height, img_width = 240, 240
batch_size = 16  # Reduced batch size for stability
learning_rate = 0.001
max_epochs = 11  # Set the number of epochs you want to run

# Data Augmentation - Simplified to reduce overhead and improve consistency
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Shuffle only training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # No shuffle for validation data
)

# Calculating exact steps for each epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Convert generators to tf.data.Dataset for stable prefetching
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, len(train_generator.class_indices)), dtype=tf.float32))
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, len(validation_generator.class_indices)), dtype=tf.float32))
).prefetch(tf.data.AUTOTUNE)

# Building the CNN model with tf.keras.layers.Input for the input shape
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Added dropout for regularization
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compiling the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Custom callback to stop training after a specified epoch and save final `.h5` model
class StopAtEpochCallback(Callback):
    def __init__(self, stop_epoch, final_model_path):
        super(StopAtEpochCallback, self).__init__()
        self.stop_epoch = stop_epoch
        self.final_model_path = final_model_path

    def on_epoch_end(self, epoch, logs=None):
        # Save the model at the end of each epoch in `.keras` format
        self.model.save(f"/content/cnn_model_epoch_{epoch + 1}.keras")
        if epoch + 1 == self.stop_epoch:
            print(f"\nStopping training at epoch {self.stop_epoch}")
            self.model.stop_training = True
            # Save the final model in `.h5` format for compatibility
            self.model.save(self.final_model_path)

# Set the callback to stop after max_epochs and save as `.h5`
stop_callback = StopAtEpochCallback(stop_epoch=max_epochs, final_model_path="/content/cnn_model_final.h5")

# Adding EarlyStopping to handle potential training anomalies
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the model with stop and early stopping callbacks
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=max_epochs,
    callbacks=[stop_callback, early_stopping]
)

# Evaluating the model
# val_loss, val_acc = model.evaluate(val_ds)
# print(f"Validation Loss: {val_loss}")
# print(f"Validation Accuracy: {val_acc}")

# Plotting accuracy and loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("Model has been trained, saved, and post-training code executed.")