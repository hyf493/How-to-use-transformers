# 利用transformers库加载bert-base-uncased模型
from transformers import AutoModel
from torch import nn
import torch
import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

checkpoint = "bert-base-chinese"

# 查看最后一层输出的特征层数
''' model = AutoModel.from_pretrained(checkpoint)
    print(model)
    
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
    '''

# 创建模型类
class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        # 连接一个全连接层完成分类
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)

    def forward(self,x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.fc(cls_vectors)
        return logits

# batch_X, batch_y = next(iter(dataloader.train_dataloader))
#
# outputs = model(batch_X)
# print(outputs.shape)


