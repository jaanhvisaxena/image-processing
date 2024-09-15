import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
from PIL import Image
from io import BytesIO

print("Starting script...")

# Define the entity_unit_map and allowed units (you can expand this based on your dataset)
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Extract allowed units from the entity_unit_map
allowed_units = sorted({unit for entity in entity_unit_map for unit in entity_unit_map[entity]})
print(f"Allowed units: {allowed_units}")

# Load train data
try:
    print("Loading train.csv...")
    train_data = pd.read_csv('dataset/train.csv')
    print("train.csv loaded successfully")
except Exception as e:
    print(f"Error loading train.csv: {e}")
    exit()

# Assuming the 'image_link' column contains the image URLs
train_data['image_filename'] = train_data['image_link'].apply(lambda x: os.path.basename(x))


# Function to download images from URLs and save them locally
def download_images(image_urls, image_filenames, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for url, filename in zip(image_urls, image_filenames):
        local_image_path = os.path.join(target_dir, filename)

        if not os.path.exists(local_image_path):  # Only download if the image doesn't exist
            try:
                response = requests.get(url, timeout=5)  # Add timeout for slow responses
                if response.status_code == 200:  # Ensure that the request was successful
                    img = Image.open(BytesIO(response.content))
                    img.verify()  # Verify that the image file is valid
                    img = Image.open(BytesIO(response.content))  # Re-open after verify (PIL requirement)
                    img.save(local_image_path)
                    print(f"Image {filename} downloaded and saved.")
                else:
                    print(f"Error downloading {url}: HTTP {response.status_code}")
            except (requests.exceptions.RequestException, Image.UnidentifiedImageError) as e:
                print(f"Error downloading {url}: {e}")


# Download images
image_urls = train_data['image_link'].tolist()
image_filenames = train_data['image_filename'].tolist()
download_images(image_urls, image_filenames, 'train_images')


# Extract numeric values and units from entity_value
def extract_value_and_unit(entity_value):
    import re
    match = re.match(r"([0-9\.]+)\s*([a-zA-Z ]+)", entity_value)
    if match:
        value, unit = match.groups()
        return float(value), unit
    return None, None


train_data['numeric_value'], train_data['unit'] = zip(*train_data['entity_value'].apply(extract_value_and_unit))

# Convert units to one-hot encoding
train_data['unit'] = train_data['unit'].astype(pd.CategoricalDtype(categories=allowed_units))
unit_labels = pd.get_dummies(train_data['unit'])


# Define a custom generator to return multi-output labels in TensorFlow-compatible format
def custom_data_generator(dataframe, batch_size, datagen, image_dir):
    def generator():
        for start in range(0, len(dataframe), batch_size):
            end = min(start + batch_size, len(dataframe))
            batch_df = dataframe[start:end]

            # Load images
            images = []
            for img_path in batch_df['image_filename']:
                img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_path), target_size=(224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img)

            images = np.array(images)

            # Load numeric and unit labels
            numeric_labels = batch_df['numeric_value'].values
            unit_labels_batch = unit_labels[start:end].values

            yield images, [np.array(numeric_labels), np.array(unit_labels_batch)]

    return generator


# Create a data generator using tf.data.Dataset from the custom generator
def create_tf_dataset(data_generator_func, batch_size, output_signature):
    dataset = tf.data.Dataset.from_generator(
        data_generator_func,
        output_signature=output_signature
    )
    dataset = dataset.batch(batch_size)
    return dataset


# Define the ImageDataGenerator (with optional augmentation)
datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values between 0 and 1
    rotation_range=20,  # Randomly rotate images by 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zooming
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill missing pixels after transformation
)

# Define the signature of the output for tf.data.Dataset
output_signature = (
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Image tensor
    (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Numeric value tensor
        tf.TensorSpec(shape=(None, len(allowed_units)), dtype=tf.float32)  # Unit one-hot encoded tensor
    )
)

batch_size = 32
train_generator = create_tf_dataset(
    custom_data_generator(train_data.sample(frac=0.8), batch_size, datagen, 'train_images'),
    batch_size=batch_size,
    output_signature=output_signature
)
val_generator = create_tf_dataset(
    custom_data_generator(train_data.sample(frac=0.2), batch_size, datagen, 'train_images'),
    batch_size=batch_size,
    output_signature=output_signature
)

# Build the ResNet50 model
try:
    print("Building ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print("ResNet50 model loaded successfully")
except Exception as e:
    print(f"Error loading ResNet50: {e}")
    exit()

x = Flatten()(base_model.output)
numeric_value_output = Dense(1, activation='linear', name='numeric_value_output')(x)
unit_output = Dense(len(allowed_units), activation='softmax', name='unit_output')(x)

# Compile the model
try:
    print("Compiling model...")
    model = Model(inputs=base_model.input, outputs=[numeric_value_output, unit_output])
    model.compile(optimizer='adam', loss={'numeric_value_output': 'mse', 'unit_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    print("Model compiled successfully")
except Exception as e:
    print(f"Error during model compilation: {e}")
    exit()

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model_checkpoint = ModelCheckpoint('best_entity_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)

# Train the model with data generators
try:
    print("Starting model training...")
    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(train_data) // batch_size

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=10,
        callbacks=[early_stopping, model_checkpoint, tensorboard],
        verbose=1
    )
    print("Training complete.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Save final model
try:
    model.save('final_entity_model.keras')
    print("Model saved as 'final_entity_model.keras'.")
except Exception as e:
    print(f"Error saving model: {e}")
