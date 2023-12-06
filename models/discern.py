import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, Config):
        super(RegressionModel, self).__init__()

        # Embedding层 for VIN
        self.embedding_vin = nn.Embedding(Config.vocab_size_vin, Config.embedding_dim)

        # Embedding层 for Combined
        self.embedding_combined = nn.Embedding(Config.vocab_size_combined, Config.embedding_dim)

        # LSTM层
        self.lstm = nn.LSTM(Config.embedding_dim, Config.hidden_size, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(Config.hidden_size, 1)  # 输出一个值

    def forward(self, vin, combined):
        # Embedding层
        embedded_vin = self.embedding_vin(vin)
        embedded_combined = self.embedding_combined(combined)

        # 将VIN和Combined的Embedding拼接
        embedded = torch.cat((embedded_vin, embedded_combined), dim=1)

        # 将Embedding的输出传递给LSTM层
        lstm_out, _ = self.lstm(embedded)

        # 取最后一个时间步的输出
        lstm_last_hidden = lstm_out[:, -1, :]

        # 通过全连接层进行回归
        output = self.fc(lstm_last_hidden)

        return output.squeeze()  # 去除维度为1的维度
