from data import NERDataset

train_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.train")
val_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.dev")
test_data = NERDataset("D:/data/china-people-daily-ner-corpus/china-people-daily-ner-corpus/example.test")

print(train_data[0])
