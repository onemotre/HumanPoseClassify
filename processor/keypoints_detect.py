import os
import yaml
import cv2
from tqdm import tqdm
from ultralytics import YOLO

import dataIO
import argument_dataset


os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
source_root = os.path.join(os.getcwd(), "assets", "videos")
data_root = os.path.join(os.getcwd(), "data")
detect_model = YOLO('yolo11m-pose')

def get_video_keypoints_data(video_path):
  '''
  generate keypoint data
  @param video_path: get keypoint labeled by YOLOv11 pre-trained model
  '''
  dataIO.create_keypoint_label_data_dir(video_path=video_path)
  
  # opencv2
  video = cv2.VideoCapture(video_path)
  total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  with tqdm(total=total_frame, desc=f"Process Video{video_path}") as pbar:
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame_count += 1

        # detect
        results = detect_model(frame, show=False, verbose=False)

        # output to data
        frame_data = {
            "file_name": f"{frame_count}.jpg",
            "pic_size":{
                "w": video.get(cv2.CAP_PROP_FRAME_WIDTH),
                "h": video.get(cv2.CAP_PROP_FRAME_HEIGHT)},
            "keypoints": []}
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
                            "confidence": float(conf)})
                    frame_data["keypoints"].append(person_data)

        cv2.imwrite(
            os.path.join(
                dataIO.get_origin_pic_dir(video_path=video_path),
                f"{frame_count}.jpg"), 
            frame)
        file_name = os.path.join(
            dataIO.get_origin_pic_dir(video_path=video_path),
            f"{frame_count}.yaml"
        )
        with open(file_name, 'w') as yaml_file:
            yaml.dump(frame_data, yaml_file, default_flow_style=False)

        # extend data
        dir_name = dataIO.get_extend_pic_dir(video_path=video_path)
        pics = argument_dataset.extended_data(frame)
        for key in pics.keys():
            cv2.imwrite(os.path.join(dir_name, f"{frame_count}-{key}.jpg"), pics[key])

        pbar.update(1)
    video.release()



if __name__ == "__main__":
    get_video_keypoints_data(f"/home/tdt/Documents/project/HumanPoseClassify/assets/videos/zz/video1.mp4")
