# 创建序列标注任务的数据集
from torch.utils.data import Dataset

categories = set()

class NERDataset(Dataset):
    # 获取文件路径
    def __init__(self, file_path):
        self.data = self.data_load(file_path)

    def data_load(self, file_path):
        Data = {}
        # 读取数据文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 数据集一行对应一个字，句子之间采用空行分割，先通过空行且分出句子，然后按行读取句子中的每一个字和对应的标签，如果标签以 B 或者 I 开头，就表示出现实体
            # 先通过空行且分出句子， 然后使用enumerate遍历句子
            for idx, line in enumerate(f.read().strip().split('\n\n')):
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    word, tag = item.split(" ")
                    sentence += word
                    if tag.startswith('B'):
                        labels.append([i, i, word, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:]) # 集合中的元素不会出现重复。
                    elif tag.startswith('I'):
                        labels[-1][1] = i # Update the end index of the last entity
                        labels[-1][2] += word # Update the entity text
                Data[idx] = {
                    'sentence': sentence,
                    'labels': labels
                }
            return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.train")
val_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.dev")
test_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.test")


id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

# print(id2label)
# print(label2id)

