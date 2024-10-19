import os

# 指定你想读取文件名的目录
directory = os.getcwd()  # 例如，我们使用当前工作目录

# 获取目录中所有文件名
filenames = os.listdir(os.path.join(directory, "data"))

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
print()
video_path = f"/home/tdt/Documents/project/HumanPoseClassify/assets/videos/zz/video1.mp4"
print(os.path.relpath(video_path, os.getcwd()))