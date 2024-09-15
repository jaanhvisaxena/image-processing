import os
from PIL import Image
import requests

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Re-download if image is corrupted
image_directory = 'train_images'
image_urls = {...}  # Dictionary of image URLs

for filename in os.listdir(image_directory):
    try:
        img = Image.open(os.path.join(image_directory, filename))
        img.verify()  # Verify image integrity
    except (OSError, IOError):
        print(f"Corrupted image found: {filename}. Re-downloading...")
        download_image(image_urls[filename], os.path.join(image_directory, filename))
