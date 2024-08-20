from torch.utils.data import Dataset
import json

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    # 创建load_data函数
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引返回数据
        return self.data[idx]


# 创建自定义数据集
train_dataset = AFQMC("D:/data/afqmc_public/train.json")
test_dataset = AFQMC("D:/data/afqmc_public/test.json")

# 查看train_dataset数据是否正常
# print(train_dataset[0])
# print(test_dataset[0])
