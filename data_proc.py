import re
import nltk
from text_proc import utterance_process

data_path = r"data/"
data_type = ["train", "valid", "test"]

glove_path = r""

word_dictionary = {}

class Word:
    def __init__(self, word_id):
        self.word_id = word_id
        self.word_vec = []

class Paragraph:
    def __init__(self, tag):
        self.tag = tag
        self.sentence_list = []

class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list
        self.word_id_list = []

    def word_proc(self):
        for word in self.word_list:
            if word in word_dictionary:
                word_id = word_dictionary[word].word_id
            else:
                word_id = len(word_dictionary) + 1
                word_dictionary[word] = Word(word_id)
            self.word_id_list.append(word_id)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def separate_paragraph(text):
    sentence_list = tokenizer.tokenize(text)
    return sentence_list

def read_data_file(file):
    for line in file:
        re_pattern = "\{\"label\"\: \"(.*?)\", \"text\"\: \"(.*?)\"}"
        res = re.match(re_pattern, line, re.S)
        tag = res.group(1)
        text = res.group(2)

        sentence_list = separate_paragraph(text)

        for sentence in sentence_list:
            sent = utterance_process(sentence)


file = open("{}{}_data".format(data_path, data_type[0]), "r", encoding="utf-8")
read_data_file(file)
