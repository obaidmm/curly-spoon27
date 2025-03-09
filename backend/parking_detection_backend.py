import os
import cv2
import argparse
import numpy as np
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Ensure output directory exists
OUTPUT_DIR = "/Users/obaidmohiuddin/Downloads/test_data/output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.txt")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    image_path = data.get("image_path")
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid or missing image path"}), 400
    
    # Load and process image
    img = Image.open(image_path)
    results = model(img)
    
    output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    detections = []
    
    with open(RESULTS_FILE, "w") as f_out:
        for result in results:
            for box in result.boxes.data:
                x_min, y_min, x_max, y_max = map(float, box[:4])  # Bounding box
                confidence = float(box[4])  # Confidence score
                class_id = int(box[5])  # Class label
                converted_class_id = class_id + 1  # Convert class IDs
                
                # Define label text
                label_text = "1" if converted_class_id == 1 else "2"
                
                # Draw bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min - 10), f"{label_text}", fill="red", font=font)
                
                # Store detection details
                detection_info = {
                    "class_id": converted_class_id,
                    "confidence": confidence,
                    "bounding_box": [x_min, y_min, x_max, y_max]
                }
                detections.append(detection_info)
                
                # Write results to file
                f_out.write(f"{os.path.basename(image_path)} {converted_class_id} {confidence:.4f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}\n")
    
    # Save processed image
    img.save(output_image_path)
    
    # Return JSON response with detection results
    return jsonify({
        "image_path": output_image_path,
        "detections": detections,
        "results_file": RESULTS_FILE
    })
    
@app.route("/download", methods=["GET"])
def download():
    image_path = request.args.get("image_path")
    
    if not image_path:
        return jsonify({"error": "Missing image path"}), 400

    # Ensure the image is in the correct output directory
    safe_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))

    # Debug: Print path to verify correctness
    print(f"Attempting to send file: {safe_path}")

    if not os.path.exists(safe_path):
        return jsonify({"error": f"File not found at {safe_path}"}), 404
    
    # Ensure the file has proper read permissions
    os.chmod(safe_path, 0o644)

    return send_file(safe_path, mimetype="image/png")



@app.route("/get_results", methods=["GET"])
def get_results():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "Results file not found"}), 404
    
    with open(RESULTS_FILE, "r") as f:
        results_data = f.readlines()
    
    return jsonify({"results": results_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)