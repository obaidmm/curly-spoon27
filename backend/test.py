import os
from ultralytics import YOLO
from PIL import Image

# Define paths
MODEL_PATH = "/Users/obaidmohiuddin/Desktop/curly-spoon27/backend/best.pt"
IMAGE_FOLDER = "/Users/obaidmohiuddin/Downloads/test_data/test_images/"
RESULTS_FILE = "/Users/obaidmohiuddin/Downloads/test_data/results/results.txt"

# Load YOLOv5 model
model = YOLO(MODEL_PATH)

# Ensure results directory exists
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

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

                    # Write results to file
                    f_out.write(f"{filename} {class_id} {confidence} {x_min} {y_min} {x_max} {y_max}\n")

print(f"✅ Detection results saved in: {RESULTS_FILE}")
