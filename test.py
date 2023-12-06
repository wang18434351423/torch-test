from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import Config
from utils.data_handle import read_json

data_list = read_json('./dataset/data.json')[0]
num_classes = len(set(entry['car_name'] for entry in data_list))
your_car_model_labels = set(entry['car_name'] for entry in data_list)


class VinDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bert-base-chinese")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vin = self.data[idx]['vin']
        car_name = self.data[idx]['car_name']

        vin_embedding = self.tokenizer(vin, return_tensors='pt')['input_ids']
        car_name_embedding = self.tokenizer(car_name, return_tensors='pt')['input_ids']

        return vin_embedding, car_name_embedding


class VinModel(nn.Module):
    def __init__(self):
        super(VinModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/bert-base-cinese")
        self.fc = nn.Linear(768 * 2, num_classes)  # Assuming num_classes is the number of car models

    def forward(self, vin_embedding, car_name_embedding):
        vin_output = self.bert(vin_embedding)['last_hidden_state'][:, 0, :]
        car_name_output = self.bert(car_name_embedding)['last_hidden_state'][:, 0, :]

        concatenated_output = torch.cat((vin_output, car_name_output), dim=1)
        output = self.fc(concatenated_output)

        return output


# Assuming data_list is your dataset
train_dataset = VinDataset(data_list)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

model = VinModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

# Training loop
num_epochs = Config.num_epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for vin_embedding, car_name_embedding in train_loader:
        optimizer.zero_grad()
        outputs = model(vin_embedding, car_name_embedding)
        loss = criterion(outputs, your_car_model_labels)  # Replace with actual car model labels
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Loss: {loss.item():.4f}")
    # Validation, testing, or other evaluation steps can be added here
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")