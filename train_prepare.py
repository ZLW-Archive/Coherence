import torch
from torch.utils.data import Dataset
from data_proc.data_proc import Paragraph

class Sentence:
    def __init__(self, seq):
        self.seq = seq
        self.seq_len = len(seq)
        self.pos_seq = torch.tensor([i for i in range(1, self.seq_len + 1)])

    def extend(self, total_length):
        tmp = torch.zeros(total_length - self.seq_len).long()
        self.seq = torch.cat([self.seq, tmp], 0)
        self.pos_seq = torch.cat([self.pos_seq, tmp], 0)

class CoDataSet(Dataset):
    def __init__(self, para_list):
        assert isinstance(para_list, list)
        assert isinstance(para_list[0], Paragraph)

        self.paragraph_num = len(para_list)
        self.paragraph_tensor_list = []
        self.paragraph_tag_list = [para.tag for para in para_list]
        self.paragraph_sentence_length_list = [[len(sent) for sent in para.sentence_list] for para in para_list]

        for para in para_list:
            assert isinstance(para, Paragraph)
            max_sentence_length = max([len(sent) for sent in para.sentence_list])
            for i, sent in enumerate(para.sentence_list):
                tmp = torch.zeros(max_sentence_length - len(sent)).long()
                sent_tensor = torch.cat([torch.Tensor(sent).long(), tmp], 0)

                if i == 0:
                    para_tensor = sent_tensor.view(1, -1)
                else:
                    para_tensor = torch.cat([para_tensor, sent_tensor.view(1, -1)])

            self.paragraph_tensor_list.append(para_tensor)

    def __getitem__(self, index):
        return self.paragraph_tensor_list[index], \
               self.paragraph_sentence_length_list[index], \
               self.paragraph_tag_list[index]

    def __len__(self):
        return self.paragraph_num

