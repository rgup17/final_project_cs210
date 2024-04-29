import os 

def download_dataset(url, file_path):
    if not os.path.exists(file_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download completed!")
    else:
        print("Dataset already downloaded.")