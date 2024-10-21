import os
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from processor import dataIO

# 定义动作类别映射
label_map = {
    'stand': 0,
    'squat': 1,
    'one-foot-stand': 2,
    'open-arms': 3,
    'arms-skimbo': 4
}

# 路径设置
os.chdir(os.path.dirname(__file__))
train_data_path = os.path.join(dataIO.pic_with_keypoints_root, "database", "train")

# 自定义数据集类
class ActionDataset(Dataset):
    def __init__(self, data_list, label_map):
        self.data_list = data_list
        self.label_map = label_map
        # 计算均值和标准差用于标准化
        all_features = []
        for sample in data_list:
            kps = sample['kps']
            features = []
            for kp in kps:
                features.extend([kp['x'], kp['y'], kp['vx'], kp['vy']])
            all_features.append(features)
        all_features = torch.tensor(all_features, dtype=torch.float32)
        self.mean = all_features.mean(dim=0)
        self.std = all_features.std(dim=0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # 提取特征
        kps = sample['kps']
        features = []
        for kp in kps:
            features.extend([kp['x'], kp['y'], kp['vx'], kp['vy']])
        features = torch.tensor(features, dtype=torch.float32)
        # 标准化
        features = (features - self.mean) / self.std
        # 获取标签
        label_str = sample['class']
        label = self.label_map[label_str]
        label = torch.tensor(label, dtype=torch.long)
        return features, label

data_list = []
train_pathes = os.listdir(os.path.join(train_data_path, "labels"))
print(os.path.join(train_data_path, "labels"))
with tqdm(total=len(train_pathes), desc="Loading data") as pbar:
    for files in train_pathes:
        with open(os.path.join(train_data_path, "labels", files), 'r') as f:
                label_data = yaml.load(f, Loader=yaml.FullLoader)
                sample = {
                    "class": label_data["class"],
                    "kps": []
                }
                for point in label_data["kps"]:
                    sample["kps"].append({
                        "x": point["x"],
                        "y": point["y"],
                        "vx": point["vx"],
                        "vy": point["vy"]
                    })
        data_list.append(sample)
        pbar.update(1)
        

# 创建数据集
dataset = ActionDataset(data_list, label_map)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义模型
class ActionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ActionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

input_size = 68
hidden_size1 = 256
hidden_size2 = 128
output_size = 5

model = ActionClassifier(input_size, hidden_size1, hidden_size2, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

l1_lambda = 1e-5
l2_lambda = 1e-4

# 训练模型
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'第 {epoch+1}/{num_epochs} 轮，损失：{epoch_loss:.4f}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    print(f'验证准确率：{val_accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'action_classifier.pth')

