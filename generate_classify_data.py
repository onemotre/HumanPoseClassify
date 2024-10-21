import os
import shutil
from tqdm import tqdm
from torch.utils.data import random_split
from processor import dataIO

os.chdir(os.path.dirname(__file__))
origin_data_root = os.path.join(dataIO.pic_with_keypoints_root, "database")

label_path, img_path = dataIO.annotate_action()

# 进行数据集划分
train_ratio = 0.8
test_ratio = 0.2

all_data = os.listdir(label_path)
train_size = int(len(all_data) * train_ratio)
test_size = len(all_data) - train_size
train_data, test_data = random_split(all_data, [train_size, test_size])

# 保存数据集划分结果
train_path = os.path.join(origin_data_root, "train")
test_path = os.path.join(origin_data_root, "test")
test_img_path = os.path.join(test_path, "images")
train_img_path = os.path.join(train_path, "images")
test_label_path = os.path.join(test_path, "labels")
train_label_path = os.path.join(train_path, "labels")
dataIO.check_dirs([train_path, test_path, test_img_path, train_img_path, test_label_path, train_label_path])

with tqdm(total=len(train_data), desc="Move train data", leave=False) as pbar:
    for data in train_data:
        shutil.copy(os.path.join(label_path, data), train_label_path)
        shutil.copy(os.path.join(img_path, data.replace("yaml", "jpg")), train_img_path)

with tqdm(total=len(test_data), desc="Move test data", leave=False) as pbar:
    for data in test_data:
        shutil.copy(os.path.join(label_path, data), test_label_path)
        shutil.copy(os.path.join(img_path, data.replace("yaml", "jpg")), test_img_path)