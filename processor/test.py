import cv2
import yaml
from ultralytics import YOLO
import numpy as np

# 加载YOLOv8n-pose模型
model = YOLO('yolo11m-pose.pt')

# 打开视频文件
video = cv2.VideoCapture('/home/tdt/Documents/project/HumanPoseClassify/assets/videos/zz/video1.mp4')

# 准备YAML输出
output_data = []

# 逐帧处理视频
frame_count = 0
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}")

    # 使用YOLOv8进行关键点检测
    results = model(frame)
    
    # 处理每一帧的结果
    frame_data = {"frame": frame_count, "keypoints": []}
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            for person_idx, person_keypoints in enumerate(keypoints):
                person_data = {"person": person_idx, "points": []}
                for kp_idx, kp in enumerate(person_keypoints):
                    x, y, conf = kp
                    person_data["points"].append({
                        "id": kp_idx,
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf)
                    })
                frame_data["keypoints"].append(person_data)
    
    output_data.append(frame_data)

# 关闭视频
video.release()

# 将数据写入YAML文件
with open('keypoints_output.yaml', 'w') as yaml_file:
    yaml.dump(output_data, yaml_file, default_flow_style=False)

print("Processing complete. Results saved to keypoints_output.yaml")