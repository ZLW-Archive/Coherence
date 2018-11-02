import numpy as np
from data_proc.data_proc import read_proc_data
from train_prepare import CoDataSet
from torch.utils.data import DataLoader

# parameter
epoch_num = 50
batch_size = 0
embedding_dim = 300
dropout_rate = 0.8

# data preparation
paragraph_list = {
    "train": read_proc_data("train"),
    "valid": read_proc_data("valid"),
    "test": read_proc_data("test")
}
paragraph_dataset = {
    "train": CoDataSet(paragraph_list["train"]),
    "valid": CoDataSet(paragraph_list["valid"]),
    "test": CoDataSet(paragraph_list["test"])
}
paragraph_loader = {
    "train": DataLoader(dataset=paragraph_dataset["train"],
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False),
    "valid": DataLoader(dataset=paragraph_dataset["valid"],
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False),
    "test": DataLoader(dataset=paragraph_dataset["test"],
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False),
}

word_vector_matrix = np.load("data/word_vector_matrix.npy")


