import os
import cv2
import scipy
from tqdm import tqdm

# 设置当前工作目录为项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
video_path = os.path.join(os.getcwd(), "assets", "videos", "video1.mp4")

# 读取视频，对每一帧进行处理
print("file path: ", video_path)
cap = cv2.VideoCapture(video_path)
frame_idx = 0

# 设置tqdm
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames) as pbar:
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      # 调整亮度
      brightness = 60
      frame_brighter = frame.copy()
      frame_brighter = cv2.add(frame_brighter, brightness)
      # rgb颜色通道添加高斯噪声
      frame_noisy = frame.copy()
      for i in range(3):
          frame_noisy[:, :, i] = scipy.ndimage.gaussian_filter(frame_noisy[:, :, i], sigma=10)
      # 图片添加高斯模糊
      frame_blur = cv2.GaussianBlur(frame, (11, 11), 0)
      # 保存处理后的图片到assets/pics目录
      save_path = os.path.join(os.getcwd(), "assets", "pics")
      if not os.path.exists(save_path):
          os.makedirs(save_path)

      frame_idx += 1
      cv2.imwrite(os.path.join(save_path, f"frame_{frame_idx}.jpg"), frame)
      cv2.imwrite(os.path.join(save_path, f"frame_{frame_idx}_brighter.jpg"), frame_brighter)
      cv2.imwrite(os.path.join(save_path, f"frame_{frame_idx}_noisy.jpg"), frame_noisy)
    
      pbar.update(1)

cap.release()
print("Done!")