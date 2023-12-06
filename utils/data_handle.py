import json
import os.path

import pandas as pd
from tqdm import tqdm


def read_json(path):
    data = list()
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            dic = json.loads(line)
            data.append(dic)
    return data


def init_data_json(data_file_path):
    file_path = '../dataset/data.xlsx'
    sheet_name = '1'
    # 读取Excel文件
    car_data = pd.read_excel(file_path, sheet_name=sheet_name)
    data_list = list()
    # 逐行读取数据
    for index, row in car_data.iterrows():
        vin = row['vin']
        car_name = row['car_name']
        car_id = row['car_id']
        data_list.append({
            'vin': vin,
            'car_name': car_name,
            'car_id': car_id
        })
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
    with open(data_file_path, 'x', encoding='utf-8') as f:
        f.write(json.dumps(data_list, ensure_ascii=False))


def handle():
    init_data_json('../dataset/data.json')
    data_list = read_json('../dataset/data.json')[0]

    # 车型名称_id 组合
    for entry in data_list:
        entry['combined'] = f"{entry['car_name']}_{entry['car_id']}"

    # vin码 映射字典
    vin_mapping = {char: idx for idx, char in enumerate(set("".join([entry['vin'] for entry in data_list])))}
    vin_mapping_file_path = 'vin_mapping.json'
    with open(vin_mapping_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(vin_mapping, ensure_ascii=False))
    max_vin_length = max(len(entry['vin']) for entry in data_list)
    vin_padding = 0  # 使用0进行填充

    # 车型名称_id 组合 映射字典
    combined_mapping = {char: idx for idx, char in enumerate(set("".join([entry['combined'] for entry in data_list])))}
    combined_mapping_file_path = 'combined_mapping.json'
    with open(combined_mapping_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(combined_mapping, ensure_ascii=False))

    max_combined_length = max(len(entry['combined']) for entry in data_list)
    combined_padding = 0  # 使用0进行填充

    for entry in data_list:
        vin_encoded = [vin_mapping[char] for char in entry.get('vin')]
        vin_encoded += [vin_padding] * (max_vin_length - len(vin_encoded))
        entry['vin_encoded'] = vin_encoded

        combined_encoded = [combined_mapping[char] for char in entry.get('combined')]
        combined_encoded += [combined_padding] * (max_combined_length - len(combined_encoded))
        entry['combined_encoded'] = combined_encoded

    total_length = len(data_list)
    train_end_index = int(total_length * 0.8)
    test_end_index = int(total_length * 0.99)

    train_data = data_list[:train_end_index]
    train_data_file_path = 'train.json'
    with open(train_data_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(train_data, ensure_ascii=False))

    test_data = data_list[train_end_index:test_end_index]
    test_data_file_path = '../dataset/test.json'
    with open(test_data_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(test_data, ensure_ascii=False))

    valid_data = data_list[test_end_index:]
    valid_data_file_path = '../dataset/valid.json'
    with open(valid_data_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(valid_data, ensure_ascii=False))


if __name__ == '__main__':
    handle()
