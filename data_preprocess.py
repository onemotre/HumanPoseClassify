import os

from processor import keypoints_detect as kps_detect
from processor import dataIO

os.chdir(os.path.dirname(__file__))
video_path = os.path.join(os.getcwd(), "assets", "videos")

process_video = [
  os.path.join(video_path, "zz", "video1.mp4"),
  os.path.join(video_path, "zz", "video2.mp4"),
  os.path.join(video_path, "lxy", "lxy.mp4")
]

for path in process_video:
  data_dir = kps_detect.get_video_keypoints_data(path)
  dataIO.format2yolo(data_dir)
  print(f"Finish {path}")

print(f"Finish data preprocess, next classify action from path:")
print(f"{dataIO.pic_with_keypoints_root}/origin, ")
print(f"to {dataIO.pic_with_keypoints_root}/classification")