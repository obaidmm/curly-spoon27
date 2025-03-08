import os
from ultralytics import YOLO

def train_yolov5():
    # Verify the data.yaml file exists
    data_yaml_path = "/Users/obaidmohiuddin/Desktop/curly-spoon27/backend/data.yaml"
    print(f"Current working directory: {os.getcwd()}")
    print(f"Resolved data.yaml path: {os.path.abspath(data_yaml_path)}")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"'{data_yaml_path}' does not exist. Please check the file path.")

    # Load a small YOLOv5 model
    model = YOLO("yolov5su.pt")  # Small ultra-lightweight model

    # Train the model with Raspberry Pi-friendly settings
    results = model.train(
        data=data_yaml_path,
        epochs=5,  # Reduced from 50 to 5
        batch=4,  # Prevents Raspberry Pi memory overload
        imgsz=640,  # Kept image size at 640x640 as requested
        device="cpu",  # Use CPU (Raspberry Pi has no GPU)
        name="parking_spot_detection_pi",
        patience=3,  # Early stopping if no improvement in 3 epochs
        optimizer="SGD",
        lr0=0.005,  # Lowered learning rate for stability
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=1,  # Shortened warmup
        warmup_momentum=0.7,
        warmup_bias_lr=0.05,
        box=0.05,
        cls=0.5,
        flipud=0.0,  # No vertical flipping
        fliplr=0.3,  # 30% chance of horizontal flip
        mosaic=0.5,  # Lowered mosaic probability (less computation)
        mixup=0.0,  # Disabled mixup (reducing compute load)
    )

    # Save the trained model
    model.export(format="onnx")  # Export model to ONNX format
    print("Training completed and model exported!")

if __name__ == "__main__":
    train_yolov5()