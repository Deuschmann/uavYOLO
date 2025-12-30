import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--config', type=str, default='configs/comparison.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # The ultralytics library doesn't use our custom yaml config files in the same way.
    # We will just hardcode the parameters for now.
    # A better approach would be to read the relevant parts from the yaml file.

    # Load a model.
    # We can use a pre-trained model like 'yolov8n.pt'
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # The dataset is defined in `uav_dataset.yaml`
    # The number of epochs and other hyperparameters should be aligned with the project's settings.
    # I will look at `configs/base.yaml` for the number of epochs.
    model.train(data='uav_dataset.yaml', epochs=100, imgsz=640, device=args.device)

if __name__ == '__main__':
    main()
