import numpy as np
from model.model import ParagraphBatchProcessor
from data_proc.read_proc_data import read_proc_data
from train_prepare import CoDataSet

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# parameter
epoch_num = 50
batch_size = 50
embedding_dim = 300
hidden_dim = 300
fc_dim = 128
dropout_rate = 0.3

# <editor-fold desc="data preparation">
paragraph_list = {
    "train": read_proc_data("train"),
    "valid": read_proc_data("valid"),
    # "test": read_proc_data("test")
}
paragraph_dataset = {
    "train": CoDataSet(paragraph_list["train"], "train"),
    "valid": CoDataSet(paragraph_list["valid"], "valid"),
    # "test": CoDataSet(paragraph_list["test"], "test")
}
# </editor-fold>

word_vector_matrix = np.load("data/word_vector_matrix.npy")

vocabulary_size = word_vector_matrix.shape[0]

model = ParagraphBatchProcessor(embedding_dim=embedding_dim,
                                hidden_dim=hidden_dim,
                                fc_dim=fc_dim,
                                vocab_size=vocabulary_size,
                                tagset_size=2,
                                word_vec_matrix=word_vector_matrix,
                                dropout=dropout_rate,
                                batch_size=batch_size,
                                attention_op="none"
                                ).cuda()

print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduce=True, size_average=True)

def train():
    print("=============================")
    print("Start Training ...")

    model.train()

    correct_predict = [0, 0]
    total_loss = 0.

    for batch_times, (paragraph_batch, sentence_length_batch, tag_batch) \
            in enumerate(paragraph_dataset["train"].loader(batch_size)):
        score_batch = model(paragraph_batch, sentence_length_batch)
        predict = torch.max(score_batch, 1)[1]

        loss = loss_func(score_batch, tag_batch)
        total_loss += loss*batch_size

        optimizer.zero_grad()
        loss.backward()

        for i in range(batch_size):
            if predict[i] == tag_batch[i]:
                correct_predict[tag_batch[i]] += 1

        if (batch_times + 1)*batch_size % 2000 == 0:
            print("{} Paragraph Finish ... loss now: {:.5f}".format((batch_times + 1)*batch_size, loss))

    return correct_predict, total_loss

def evaluate(tag):
    print("=============================")
    print("Start Evaluate ...")

    model.eval()

    correct_predict = [0, 0]
    total_loss = 0.

    for batch_times, (paragraph_batch, sentence_length_batch, tag_batch) \
            in enumerate(paragraph_dataset[tag].loader(batch_size)):
        if batch_times % 10000 == 0:
            print("{} Paragraph Finish ...".format(batch_times * batch_size))

        score_batch = model(paragraph_batch, sentence_length_batch)
        predict = torch.max(score_batch, 1)[1]

        loss = loss_func(score_batch, tag_batch)
        total_loss += loss*batch_size

        for i in range(batch_size):
            if predict[i] == tag_batch[i]:
                correct_predict[tag_batch[i]] += 1

    return correct_predict, total_loss

def print_info(tag, correct_predict, total_loss):
    print("{}: CO: {}/{}; UN: {}/{}".format(tag, correct_predict[1], paragraph_dataset[tag].paragraph_tag_num[1],
                                            correct_predict[0], paragraph_dataset[tag].paragraph_tag_num[0]))
    print("{}: ACC: {:.5}".format(tag, (correct_predict[0]+correct_predict[1])/len(paragraph_dataset[tag])))
    print("{}: LOSS: {:.5}".format(tag, total_loss))

for epoch in range(epoch_num):
    train_correct_predict, train_total_loss = train()
    # evaluate_correct_predict, evaluate_total_loss = evaluate("valid")

    print_info("train", train_correct_predict, train_total_loss)
    # print_info("valid", evaluate_correct_predict, evaluate_total_loss)


