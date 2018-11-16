import random
import numpy as np
from tqdm import tqdm


class Paragraph:
    def __init__(self, source, tag):
        self.source = source
        self.tag = tag
        self.sentence_list = []


def read_proc_data(source, para_wise=False, sample=-1, pos_need=False):
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

        para_category[0] = para_category[0][:sample // 2]
        para_category[1] = para_category[1][:sample // 2]

    para_list = para_category[0] + para_category[1]

    pos_list = []
    if pos_need:
        for para in para_list:
            cur = []
            for sent in para.sentence_list:
                if len(cur) == 0:
                    cur.append(len(sent)-1)
                else:
                    cur.append(len(sent)+cur[-1])
            pos_list.append(cur)

    if para_wise:
        para_wise_list = [[], []]
        for para in para_list:
            cat = []
            for sent in para.sentence_list:
                cat += sent
            para_wise_list[para.tag].append({"paragraph": cat, "tag": para.tag})

        para_list = para_wise_list[0] + para_wise_list[1]

    if pos_need:
        return para_list, pos_list

    return para_list
