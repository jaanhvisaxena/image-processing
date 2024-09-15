import os
import numpy as np
from PIL import Image, ImageFile
import requests

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Define the preprocess_image function
def preprocess_image(file_path):
    """Load an image file, resize it to 224x224, and normalize it."""
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')  # Ensure image has 3 channels (RGB)
            img = img.resize((224, 224))  # Resize to 224x224 pixels
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None


# Define the download_image function
def download_image(url, save_path):
    """Download an image from a URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Image downloaded successfully: {save_path}")
        else:
            print(f"Failed to download image: {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {url}: {e}")


# Function to load and process images in batches
def load_images_in_batches(image_directory, image_urls=None, batch_size=100):
    """Load images in batches, ensuring consistent size and shape."""
    images = []
    files = os.listdir(image_directory)
    total_files = len(files)
    for index, file_name in enumerate(files):
        file_path = os.path.join(image_directory, file_name)
        img = preprocess_image(file_path)

        # Print image shape for debugging
        if img is not None:
            print(f"Image {file_name} loaded with shape: {img.shape}")

        # Ensure the image is not None and has the expected shape
        if img is not None and img.shape == (224, 224, 3):  # Ensure consistent shape
            images.append(img)
        else:
            print(f"Skipping image {file_name} due to inconsistent shape or None value")

        if len(images) >= batch_size:
            yield np.array(images)
            images = []  # Reset for the next batch

    if images:  # Yield any remaining images if they don't fill a complete batch
        yield np.array(images)
