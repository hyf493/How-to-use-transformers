from tqdm.auto import tqdm
import torch
from transformers import AdamW
import model
from transformers import get_scheduler
import dataloader
from torch import nn

train_dataloader = dataloader.train_dataloader
test_dataloader = dataloader.test_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构造训练函数
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

# 构造测试函数
def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")

model = model.BertForPairwiseCLS().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

total_loss = 0.
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    test_loop(test_dataloader, model, mode='Valid')
print("Done!")