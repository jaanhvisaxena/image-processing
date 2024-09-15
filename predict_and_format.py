import pandas as pd
from tensorflow.keras.models import load_model
from preprocess_images import preprocess_image
import os
import numpy as np

# Load the test dataset
test_data = pd.read_csv('dataset/test.csv')

# Load the trained model
model = load_model('entity_model.h5')

# Preprocess test images
test_images = []
for file in os.listdir('test_images'):
    img = preprocess_image(os.path.join('test_images', file))
    test_images.append(np.array(img))
test_images = np.array(test_images)

# Make predictions
numeric_preds, unit_preds = model.predict(test_images)

# Convert unit_preds from one-hot encoding to unit labels
allowed_units = ['gram', 'kg', 'cm', 'inch', ...]  # Same as in train_model.py
unit_labels = [allowed_units[np.argmax(unit)] for unit in unit_preds]

# Format predictions
def format_prediction(value, unit):
    return f"{value:.2f} {unit}"

formatted_predictions = [format_prediction(val, unit) for val, unit in zip(numeric_preds, unit_labels)]

# Add predictions to the test dataset
test_data['prediction'] = formatted_predictions

# Save the output in the required format
test_data[['index', 'prediction']].to_csv('test_out.csv', index=False)
