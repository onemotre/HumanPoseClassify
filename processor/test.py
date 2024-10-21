import os

# 指定你想读取文件名的目录
dir_name = os.path.join(os.getcwd(), "data", "pic_keypoints_label",
                        "lxy", "classification", "database", "labels")
for files in os.listdir(dir_name):
    print(files)