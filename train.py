import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from models.discern import RegressionModel
from utils.data_load import get_train_DataLoader, get_valid_DataLoader

# 定义模型
model = RegressionModel(Config)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=Config.learning_rate)

# 设定训练的设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 数据加载器
train_dl = get_train_DataLoader(Config)
valid_dl = get_valid_DataLoader(Config)

# 训练循环
num_epochs = Config.num_epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for vin, combined, regression_label in train_dl:
        vin, combined, regression_label = vin.to(device), combined.to(device), regression_label.to(device)

        optimizer.zero_grad()
        outputs = model(vin, combined)

        # 计算损失
        loss = criterion(outputs, regression_label)  # 替换为实际的回归标签
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dl)
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        total_mse = 0.0

        for vin, combined in valid_dl:
            vin, combined, regression_label = vin.to(device), combined.to(device), regression_label.to(device)

            outputs = model(vin, combined)

            # 计算损失
            mse = criterion(outputs, regression_label)  # 替换为实际的回归标签
            total_mse += mse.item()

        average_mse = total_mse / len(valid_dl)
        print(f"Validation MSE: {average_mse:.4f}")
