from tqdm import tqdm

class Paragraph:
    def __init__(self, source, tag):
        self.source = source
        self.tag = tag
        self.sentence_list = []

def read_proc_data(source):
    file_path = r"data/{}_data_proc".format(source)
    file = open(file_path, "r", encoding="utf-8")
    para_num = int(file.readline())

    para_list = []

    for _ in tqdm(range(para_num), desc="{} reading".format(source)):
        tag = int(file.readline())
        sent_num = int(file.readline())
        para_obj = Paragraph(source, tag)
        for _ in range(sent_num):
            sent = file.readline()
            word_id_list = list(map(int, sent.split()))
            para_obj.sentence_list.append(word_id_list)
        para_list.append(para_obj)

    return para_list
