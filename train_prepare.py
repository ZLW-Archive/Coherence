import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from data_proc.read_proc_data import Paragraph

MAX_SENTENCE_NUM_IN_PARAGRAPH = 57

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
    def __init__(self, para_list, source, shuffle=False):
        assert isinstance(para_list, list)
        assert isinstance(para_list[0], Paragraph)

        self.paragraph_num = len(para_list)
        self.paragraph_tag_num = [0, 0]
        self.paragraph_tensor_list = []
        self.paragraph_tag_list = [para.tag for para in para_list]

        self.paragraph_sentence_length_list = \
            [torch.Tensor([len(sent) for sent in para.sentence_list]
                          * (MAX_SENTENCE_NUM_IN_PARAGRAPH // len(para.sentence_list))
                          # + [0 for _ in range(MAX_SENTENCE_NUM_IN_PARAGRAPH % len(para.sentence_list))]
                          ).int()
             for para in para_list]

        for i in tqdm(range(self.paragraph_num), desc="{} dataset".format(source)):
            para = para_list[i]
            assert isinstance(para, Paragraph)

            self.paragraph_tag_num[int(para.tag)] += 1

            max_sentence_length = max([len(sent) for sent in para.sentence_list])
            for j, sent in enumerate(para.sentence_list):
                tmp = torch.zeros(max_sentence_length - len(sent)).long()
                sent_tensor = torch.cat([torch.Tensor(sent).long(), tmp], 0)

                if j == 0:
                    para_tensor = sent_tensor.view(1, -1)
                else:
                    para_tensor = torch.cat([para_tensor, sent_tensor.view(1, -1)])

            # proc extend
            extend_times = MAX_SENTENCE_NUM_IN_PARAGRAPH // para_tensor.shape[0]
            rest_space = MAX_SENTENCE_NUM_IN_PARAGRAPH % para_tensor.shape[0]
            para_tensor_buff = para_tensor
            for _ in range(extend_times - 1):
                para_tensor = torch.cat([para_tensor, para_tensor_buff], 0)

            # para_tensor = torch.cat([para_tensor, torch.zeros([rest_space, para_tensor.shape[1]]).long()], 0)

            self.paragraph_tensor_list.append(para_tensor)

        self.make_cuda()

        if shuffle:
            idx = random.sample(range(self.paragraph_num), self.paragraph_num)
            self.paragraph_tensor_list = self.paragraph_tensor_list[idx]
            self.paragraph_tag_list = self.paragraph_tag_list[idx]
            self.paragraph_sentence_length_list = self.paragraph_sentence_length_list[idx]

    def make_cuda(self):
        for i in range(self.paragraph_num):
            self.paragraph_tensor_list[i] = self.paragraph_tensor_list[i].cuda()
            self.paragraph_sentence_length_list[i] = self.paragraph_sentence_length_list[i].cuda()

    def __getitem__(self, index):
        return self.paragraph_tensor_list[index], \
               self.paragraph_sentence_length_list[index], \
               torch.Tensor(self.paragraph_tag_list[index]).long().cuda()

    def __len__(self):
        return self.paragraph_num

    def loader(self, batch_size):
        times = self.paragraph_num // batch_size
        rest = self.paragraph_num % batch_size

        for i in range(times):
            yield self.paragraph_tensor_list[i: i+batch_size], \
                  self.paragraph_sentence_length_list[i: i+batch_size], \
                  torch.Tensor(self.paragraph_tag_list[i: i+batch_size]).long().cuda()
