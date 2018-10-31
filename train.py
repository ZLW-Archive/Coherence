import numpy as np
from read_proc_data import read_proc_data

paragraph_list = {
    "train": read_proc_data("train"),
    "valid": read_proc_data("valid"),
    "test": read_proc_data("test")
}

word_vector_matrix = np.load("data/word_vector_matrix.npy")

