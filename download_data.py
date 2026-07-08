from roboflow import Roboflow
import os

def download_dataset():
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="amOahArVnW8S54jlgKwh")
    project = rf.workspace("lemi-debele").project("aircraft-surface-damage")
    version = project.version(25)
    
    # Download dataset directly into the current directory
    dataset = version.download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Save the location to a file so we know exactly where it is for data_local.yaml
    with open("dataset_location.txt", "w") as f:
        f.write(dataset.location)

if __name__ == "__main__":
    download_dataset()
