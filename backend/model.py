import os
from ultralytics import YOLO

def train_yolov5():
    # Step 1: Verify the data.yaml file exists
    data_yaml_path = "/Users/obaidmohiuddin/Desktop/curly-spoon27/backend/data.yaml"  # Path to your data.yaml file
    print(f"Current working directory: {os.getcwd()}")
    print(f"Resolved data.yaml path: {os.path.abspath(data_yaml_path)}")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"'{data_yaml_path}' does not exist. Please check the file path.")

    # Step 2: Load a pre-trained YOLOv5 model
    model = YOLO("yolov5su.pt")  # Load the small YOLOv5 model

    # Step 3: Train the model
    results = model.train(
        data=data_yaml_path,  # Path to your data.yaml file
        epochs=50,  # Number of training epochs
        batch=16,  # Batch size (adjust based on your GPU memory)
        imgsz=640,  # Input image size (YOLOv5 resizes images to 640x640 by default)
        device="cpu",  # Use CPU ("0" for GPU, "cpu" for CPU)
        name="parking_spot_detection",  # Name of the training run
        patience=10,  # Early stopping if no improvement for 10 epochs
        optimizer="SGD",  # Optimizer (SGD, Adam, etc.)
        lr0=0.01,  # Initial learning rate
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias learning rate
        box=0.05,  # Box loss gain
        cls=0.5,  # Class loss gain
        hsv_h=0.015,  # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # Image HSV-Value augmentation (fraction)
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=0.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        mosaic=1.0,  # Image mosaic (probability)
        mixup=0.0,  # Image mixup (probability)
    )

    # Step 4: Save the trained model
    model.export(format="onnx")  # Export the model to ONNX format
    print("Training completed and model exported!")

if __name__ == "__main__":
    train_yolov5()