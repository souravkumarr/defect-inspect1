import os
import sys
import shutil
import torch
from ultralytics import YOLO

def main():
    print("Starting YOLOv8 training on local aircraft surface damage dataset...")
    
    # Check if dataset yaml exists
    yaml_path = os.path.abspath("data_local.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset configuration file not found at {yaml_path}")
        sys.exit(1)
        
    print(f"Loading dataset configuration from: {yaml_path}")
    
    # Load pretrained YOLOv8n model
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # Train the model
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training YOLOv8n for {epochs} epochs on {device.upper()}...")
    print("Training on 100% of dataset at 640px resolution with advanced augmentations.")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            fraction=1.0,  # Train on 100% of the dataset
            workers=0,     # Prevent overhead of multi-threaded data loading
            device=device, # Auto-select GPU if available, else CPU
            project="aircraft_damage",
            name="yolov8_train",
            patience=15,   # Early stopping
            mosaic=1.0,    # Data augmentation
            mixup=0.2,     # Data augmentation
            degrees=15.0,  # Rotation augmentation
            hsv_s=0.5      # Color augmentation
        )
        print("Training completed successfully!")
        
        # Get path of best weights
        best_weights = os.path.join("aircraft_damage", "yolov8_train", "weights", "best.pt")
        if os.path.exists(best_weights):
            print(f"Best weights saved to: {best_weights}")
            # Copy to root directory for easy access
            shutil.copy(best_weights, "best.pt")
            print("Copied best weights to root directory as 'best.pt'")
        else:
            print("Warning: Could not locate trained weights file 'best.pt'.")
            
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
