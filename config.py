class Config():
    train_path = './dataset/train.json'  # 训练数据存放路径
    valid_path = './dataset/valid.json'  # 验证数据存放路径
    test_path = './dataset/test.json'  # 测试数据存放路径
    vin_mapping_path = './dataset/vin_mapping.json'  # vin映射文件存放路径
    combined_mapping_path = './dataset/combined_mapping.json'  # 车型名称和id映射文件存放路径

    batch_size = 4  # 每个训练批有4个样本
    num_epochs = 10  # 共在数据上训练10轮
    learning_rate = 0.01  # 优化器的学习率
    weight_decay = 1e-6  # 优化器的权重衰减
    hidden_size = 64  # LSTM 层的隐藏层大小
    embedding_dim = 100  # 它表示每个字符的嵌入向量的大小
    vocab_size_vin = 33
    vocab_size_combined = 778
