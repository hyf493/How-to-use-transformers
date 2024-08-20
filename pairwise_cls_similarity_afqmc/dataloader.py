from torch.utils.data import DataLoader
import data
from transformers import AutoTokenizer
import torch

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 创建批处理collote_fn函数，然后对没饿过batch中的所有句子对进行编码，同时把标签转换成张量
def collote_fn(bach_samples):
    batch_sentence1, batch_sentence2 = [], []
    batch_label = []
    for batch_sample in bach_samples:
        batch_sentence1.append(batch_sample["sentence1"])
        batch_sentence2.append(batch_sample["sentence2"])
        batch_label.append(int(batch_sample["label"]))
    X = tokenizer(batch_sentence1, batch_sentence2, padding=True, truncation=True, return_tensors="pt")
    y = torch.tensor(batch_label)
    return X, y

train_dataset = data.train_dataset
batch_size = 4
# 通过dataloader按批次加载数据
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)

test_dataset = data.test_dataset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collote_fn)

# batch_X, batch_y = next(iter(train_dataloader))
# print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
# print('batch_y shape:', batch_y.shape)
# print(batch_X)
# print(batch_y)