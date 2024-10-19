import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import yaml

labeel_map = {
    "stand": 0,
    "squat": 1,
    "one-foot-stand": 2,
    "open-arms": 3,
    "arms-skimbo": 4
}

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
    

class ActionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ActionClassifier, self).__init__()
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.5)
        # 第一个隐藏层到第二个隐藏层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.5)
        # 输出层
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # x 的形状为 (batch_size, 68)
        # 重新组织为 (batch_size, 17, 4)
        x = x.view(-1, 17, 4)
        # 池化操作，例如对每个关键点的特征进行平均
        x = torch.mean(x, dim=2)  # (batch_size, 17)
        # 展平
        x = x.view(-1, 17)
        # 连接到全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        # 不使用 Softmax，直接输出 logits
        return x  # 输出形状为 (batch_size, output_size)