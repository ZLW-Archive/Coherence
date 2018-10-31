import re
import nltk
import numpy as np
from text_proc import utterance_process

data_path = r"data/"
data_type = ["train", "valid", "test"]
# data_type = ["try"]

data_file_path = {t: "{}{}_data".format(data_path, t)
                  for t in data_type}
data_file = {t: open(data_file_path[t], "r", encoding="utf-8")
             for t in data_type}

save_file_path = {t: "{}{}_data_proc".format(data_path, t)
                  for t in data_type}
save_file = {t: open(save_file_path[t], "w", encoding="utf-8")
             for t in data_type}

paragraph_list = {t: [] for t in data_type}

glove_path = r"D:\Documents\Code_Python\WordVector_Folder\glove.840B.300d.txt"

word_dictionary = {}

# <editor-fold desc="Function and Class">
class Paragraph:
    def __init__(self, source, tag):
        self.source = source
        self.tag = tag
        self.sentence_list = []

    def write_paragraph(self, file):
        file.write("{}\n".format(self.tag))
        file.write("{}\n".format(len(self.sentence_list)))
        for sent in self.sentence_list:
            sent.write_sentence(file)

class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list
        self.word_id_list = []

    def word_proc(self):
        for word in self.word_list:
            if word in word_dictionary:
                word_id = word_dictionary[word]
            else:
                word_id = len(word_dictionary) + 1
                word_dictionary[word] = word_id
            self.word_id_list.append(word_id)

    def write_sentence(self, file):
        for word in self.word_id_list:
            file.write("{} ".format(word))
        file.write("\n")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def separate_paragraph(text):
    sentence_list = tokenizer.tokenize(text)
    return sentence_list

def read_data_file(t, file, para_list):
    print("Start to read {}".format(t))
    for i, line in enumerate(file):
        if i % 500 == 0:
            print("{}: Read Data finish {} ...".format(t, i))

        re_pattern = "\{\"label\"\: \"(.*?)\", \"text\"\: \"(.*?)\"}"
        res = re.match(re_pattern, line, re.S)
        tag = res.group(1)
        text = res.group(2)

        para_obj = Paragraph(t, tag)
        sentence_list = separate_paragraph(text)

        for sentence in sentence_list:
            sent = utterance_process(sentence)
            sent_obj = Sentence(sent)
            sent_obj.word_proc()
            para_obj.sentence_list.append(sent_obj)

        para_list.append(para_obj)
    print("Finish read {}\n".format(t))

def build_word_vector_matrix(word_vec_path, np_matrix):
    print("Start Build Word Vec ...")
    word_vec_file = open(word_vec_path, "r", encoding="utf-8")
    found_num = 0
    for num, line in enumerate(word_vec_file):
        if num % 100000 == 0:
            print("line {} ...".format(num))
        word, vec = line.split(' ', 1)
        if word in word_dictionary:
            word_id = word_dictionary[word]
            np_matrix[word_id] = np.array(list(map(float, vec.split())))
            found_num += 1
    print("Found {}/{} words with glove vectors\n".format(found_num, len(word_dictionary)))
# </editor-fold>

for t in data_type:
    read_data_file(t, data_file[t], paragraph_list[t])

word_num = len(word_dictionary) + 1
embedding_dim = 300

word_vector_matrix = 0.5 * np.random.random_sample((word_num, embedding_dim)) - 0.25
word_vector_matrix[0] = 0
build_word_vector_matrix(glove_path, word_vector_matrix)

# Save
np.save(r"data/word_vector_matrix.npy", word_vector_matrix)
for t in data_type:
    print("Start to save type: {}".format(t))
    save_file[t].write("{}\n".format(len(paragraph_list[t])))
    for i, para in enumerate(paragraph_list[t]):
        if i % 500 == 0:
            print("{}: finish para {}".format(t, i))
        para.write_paragraph(save_file[t])
    print("Finish save type: {}\n".format(t))
