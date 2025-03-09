import os
import argparse
from ultralytics import YOLO
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run YOLO model on images and save results.")
parser.add_argument("--image_folder", type=str, required=True, help="Path to the input images directory.")
parser.add_argument("--results_folder", type=str, required=True, help="Path to save results.")
args = parser.parse_args()

# Define paths dynamically
MODEL_PATH = "best.pt"  # Ensure the model is in the same directory as test.py
IMAGE_FOLDER = args.image_folder
RESULTS_FILE = os.path.join(args.results_folder, "results.txt")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure results directory exists
os.makedirs(args.results_folder, exist_ok=True)

# Open results.txt to store detections
with open(RESULTS_FILE, "w") as f_out:
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(IMAGE_FOLDER, filename)

            # Load and process image
            img = Image.open(image_path)

            # Run inference
            results = model(img)

            # Extract detection data
            for result in results:
                for box in result.boxes.data:
                    x_min, y_min, x_max, y_max = map(float, box[:4])  # Bounding box
                    confidence = float(box[4])  # Confidence score
                    class_id = int(box[5])  # Class label

                    # Convert YOLO class IDs (0 → 1, 1 → 2)
                    converted_class_id = class_id + 1

                    # Write results to file
                    f_out.write(f"{filename} {converted_class_id} {confidence:.4f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}\n")

print(f"✅ Detection results saved in: {RESULTS_FILE}")
