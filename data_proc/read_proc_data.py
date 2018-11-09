import random
import numpy as np
from tqdm import tqdm

class Paragraph:
    def __init__(self, source, tag):
        self.source = source
        self.tag = tag
        self.sentence_list = []

def read_proc_data(source, para_wise=False, sample=-1):
    file_path = r"data/{}_data_proc".format(source)
    file = open(file_path, "r", encoding="utf-8")
    para_num = int(file.readline())

    para_category = [[], []]

    for _ in tqdm(range(para_num), desc="{} reading".format(source)):
        tag = int(file.readline())
        sent_num = int(file.readline())
        para_obj = Paragraph(source, tag)
        for _ in range(sent_num):
            sent = file.readline()
            word_id_list = list(map(int, sent.split()))
            para_obj.sentence_list.append(word_id_list)
        para_category[tag].append(para_obj)

    if sample != -1:
        random.Random().shuffle(para_category[0])
        random.Random().shuffle(para_category[1])

        para_category[0] = para_category[0][:sample//2]
        para_category[1] = para_category[1][:sample//2]

    if para_wise:
        para_wise_list = [[], []]
        for para in para_category[0]:
            cat = []
            for sent in para.sentence_list:
                cat += sent
            para_wise_list[0].append({"paragraph": cat, "tag": 0})
        for para in para_category[1]:
            cat = []
            for sent in para.sentence_list:
                cat += sent
            para_wise_list[1].append({"paragraph": cat, "tag": 1})

        return para_wise_list[0] + para_wise_list[1]

    return para_category[0] + para_category[1]
