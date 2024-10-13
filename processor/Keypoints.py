import os
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
source_path = os.path.join(os.getcwd(), "assets", "videos", "video1.mp4")
save_path = os.path.join(os.getcwd(), "assets", "result")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load YOLOv11 model
model = YOLO('yolo11m-pose')

# RUN
result = model(source=source_path, show=True, save=True, save_dir=save_path, device='0')