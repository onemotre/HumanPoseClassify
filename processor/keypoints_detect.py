import os
import yaml
import cv2
import time
import torch
import comet_ml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


import dataIO
import argument_dataset
import EKF

# basic settings
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
source_root = os.path.join(os.getcwd(), "assets", "videos")
data_root = os.path.join(os.getcwd(), "data")
detect_model = YOLO('yolo11x-pose')
pretrained_model = YOLO("yolo11m-pose.pt")

# Kalman Filter
Q = torch.eye(4) * 0.2
R = torch.eye(2) * 0.01

pointKF = []

def f(x, dt):
    '''
    @param x: state vector (x, y, vx, vy)
    @param dt: time interval
    '''
    return torch.tensor([
        x[0] + x[2] * dt,
        x[1] + x[3] * dt,
        x[2],
        x[3]
    ])

def h(x):
    return torch.tensor([
        x[0],
        x[1]
    ])

def kalman_filter(idx, x, y, dt):
    if idx >= len(pointKF):
        # init new EKF
        ekf = EKF.EKF(x, y, f, h, Q, R)
        

    

def get_video_keypoints_data(video_path):
  '''
  generate keypoint data
  @param video_path: get keypoint labeled by YOLOv11 pre-trained model
  @return the directory of the origin picture
  '''
  dataIO.create_keypoint_label_data_dir(video_path=video_path)
  
  # opencv2
  video = cv2.VideoCapture(video_path)
  total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  total_inference_time = 0
  total_IO_time = 0
  with tqdm(total=total_frame, desc=f"Process {os.path.relpath(video_path, os.getcwd())}") as pbar:
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame_count += 1

        # detect
        inference_start_time = time.time()
        results = detect_model(frame, show=False, verbose=False, device='0')
        inferece_time = time.time() - inference_start_time
        total_inference_time += inferece_time
        avg_inference_time = total_inference_time / frame_count

        # output to data
        io_start_time = time.time()
        frame_data = {
            "file_name": f"{frame_count}.jpg",
            "pic_size":{
                "w": video.get(cv2.CAP_PROP_FRAME_WIDTH),
                "h": video.get(cv2.CAP_PROP_FRAME_HEIGHT)},
            "keypoints": [],
            "boxes":[]}
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_idx, (person_keypoints, person_keypoints_xyn) in enumerate(zip(keypoints, keypoints_xyn)):
                    person_data = {"person": person_idx, "points": []}
                    for kp_idx, (kp, kp_xyn) in enumerate(zip(person_keypoints, person_keypoints_xyn)):
                        x, y, conf = kp
                        xn, yn = kp_xyn
                        person_data["points"].append({
                            "class": kp_idx,
                            "x": float(x),
                            "y": float(y),
                            "xn": float(xn),
                            "yn": float(yn),
                            "confidence": float(conf)})
                    frame_data["keypoints"].append(person_data)
                boxes = result.boxes.xywh.cpu().numpy()
                for box_idx, box_xywh in enumerate(boxes):
                    xb, yb, wb, hb = box_xywh
                    frame_data["boxes"].append({
                        "box_id": box_idx,
                        "x": float(xb),
                        "y": float(yb),
                        "w": float(wb),
                        "h": float(hb)
                    })

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
            cv2.imwrite(os.path.join(dir_name, f"{frame_count}_{key}.jpg"), pics[key])
        
        io_time = time.time() - io_start_time
        total_IO_time += io_time
        avg_IO_time = total_IO_time / frame_count

        pbar.set_postfix({f"YOLO": f"{avg_inference_time * 1000:.2f}ms", 
                          f"IO": f"{avg_IO_time * 1000:.2f}ms"})
        pbar.update(1)
    pbar.close()
    video.release()
    return os.path.dirname(dataIO.get_origin_pic_dir(video_path=video_path))


def train_keypoint_modle(data_set, epochs=10, img_size=640):
    '''
    train keypoint model by generated dataset
    split algorithm: K Fold
    @param data_set: path of the set, eg: ./data/pic_keypoints_label/video1
    '''
    experiment = comet_ml.Experiment(
        api_key="65RLteBNAj1yCiKoB5m3ykFCG",
        project_name="human-pose-classify",
        workspace="onemotre"
    )

    kfolds = dataIO.split_yolo_train_val(data_set, K=0)
    fold_results = []
    with tqdm(total=len(kfolds), desc="Training K-Folds", position=0, leave=True) as pbar:
        for fold, kfold in enumerate(kfolds):
            # train
            results = pretrained_model.train(
                data=os.path.join(kfold, "coco8-pose.yaml"),
                epochs=epochs,
                imgsz=img_size,
                batch=16,
                iou = 0.6,

                device='0',
                workers=8,
                save=True,
                save_period=1,
                project=os.path.join(kfold, "runs"))
            
            for epoch, metrics in results.items():
                experiment.log_metrics(metrics, step=epoch)
            
            # validate
            val_results = pretrained_model.val(
                data=os.path.join(kfold, "coco8-pose.yaml"),
                batch=16,
                device='0',
                project=os.path.join(kfold, "runs"),
            )
            experiment.log_metrics(val_results, step=epochs)
            
            # save
            with open(os.path.join(kfold, "results.txt"), 'w') as f:
                f.write(val_results)
            
            fold_results.append(val_results)
            experiment.end()
            pbar.update(1)
    
    mean_results = np.mean(fold_results, axis=0)
    print(f"Mean validation results across {K} folds: {mean_results}")
    
    # Save mean results
    with open(os.path.join(data_set, "mean_results.txt"), 'w') as f:
        f.write(f"Mean validation results across {K} folds: {mean_results}")


if __name__ == "__main__":
    data_dir = get_video_keypoints_data(f"/home/tdt/Documents/project/HumanPoseClassify/assets/videos/lxy/lxy.mp4")
    # dataIO.format2yolo(data_dir)
    # train_keypoint_modle("/home/tdt/Documents/project/HumanPoseClassify/data/pic_keypoints_label/video1")
