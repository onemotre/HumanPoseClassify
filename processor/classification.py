import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AdaptiveRandomPool(nn.Module):
    def __init__(self, output_size, pooling_type='max'):
        super(AdaptiveRandomPool, self).__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        
        # 计算需要的池化窗口大小
        stride = seq_len // self.output_size
        kernel_size = seq_len - (self.output_size - 1) * stride
        
        # 添加随机偏移
        offset = torch.randint(0, stride, (batch_size, channels, 1), device=x.device)
        x = F.pad(x, (0, stride))  # 在序列末尾填充，以允许偏移
        x = x.view(batch_size, channels, -1, stride)
        x = x[:, :, :, offset.squeeze(-1)]
        x = x.view(batch_size, channels, -1)
        
        # 应用池化
        if self.pooling_type == 'max':
            x = F.max_pool1d(x, kernel_size=kernel_size, stride=stride)
        elif self.pooling_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError("Unsupported pooling type")
        
        return x

class ActionClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ActionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.random_pool1 = AdaptiveRandomPool(output_size=8, pooling_type='max')
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.random_pool2 = AdaptiveRandomPool(output_size=4, pooling_type='max')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 变换维度以适应Conv1d
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.random_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.random_pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# 评估函数
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

# 主程序
def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 定义超参数
    num_classes = 5
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 32

    # 创建模型
    model = ActionClassifier(num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 假设我们有训练和测试数据加载器
    # 这里你需要根据实际情况创建 DataLoader
    # train_loader = ...
    # test_loader = ...

    # 训练模型
    train(model, train_loader, criterion, optimizer, num_epochs)

    # 评估模型
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()