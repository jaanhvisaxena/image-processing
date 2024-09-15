import requests
import os

def download_images(url, save_directory):
    try:
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Extract the image filename from the URL
        filename = os.path.basename(url)
        filepath = os.path.join(save_directory, filename)

        # Download and save the image
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
