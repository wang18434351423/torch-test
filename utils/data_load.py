import torch
from torch.utils.data import DataLoader, Dataset
from utils.data_handle import read_json


class CustomDataset(Dataset):
    def __init__(self, data, Config, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vin = torch.tensor(self.data[idx]['vin_encoded'], dtype=torch.long)
        combined = torch.tensor(self.data[idx]['combined_encoded'], dtype=torch.long)
        regression_label = torch.tensor(float(self.data[idx]['car_id']), dtype=torch.float32)  # 使用car_id作为回归标签

        if self.transform:
            vin, combined = self.transform(vin, combined)

        return vin, combined, regression_label


# 返回训练 DataLoader
def get_train_DataLoader(Config):
    train_dl = DataLoader(CustomDataset(read_json(Config.train_path)[0], Config),
                          batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('训练数据加载完成')
    return train_dl


# 返回测试数据 DataLoader
def get_test_DataLoader(Config):
    test_dl = DataLoader(CustomDataset(read_json(Config.test_path)[0], Config),
                         batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('测试数据加载完成')
    return test_dl


# 返回验证数据 DataLoader
def get_valid_DataLoader(Config):
    valid_dl = DataLoader(CustomDataset(read_json(Config.valid_path)[0], Config),
                          batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('验证数据加载完成')
    return valid_dl
