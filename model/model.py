import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.Attention import ScaledDotProductAttentionBatch

MAX_SENTENCE_NUM_IN_PARAGRAPH = 57

class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix,
                 attention_op="none"):
        super(SentenceEncoder, self).__init__()
        self.adj_attention_op = attention_op

        # word level
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.bilstm_layer = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    bidirectional=True,
                                    batch_first=True)

        # self.attention_layer = ScaledDotProductAttentionBatch(model_dim=2*embedding_dim)

    def forward(self, sentence_list, sentence_length_list):
        max_len = int(sentence_length_list.max())
        sentence = sentence_list[:, 0:max_len]

        # get word embedding
        embeds = self.embedding_layer(sentence)

        # sort
        sentence_length_list, indices = torch.sort(sentence_length_list, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)

        embeds = embeds[indices]

        embeds = pack_padded_sequence(embeds, sentence_length_list.cpu().numpy(), batch_first=True)
        lstm_out, _ = self.bilstm_layer(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # unsort
        lstm_out = lstm_out[desorted_indices]

        avg_k = lstm_out.shape[1]
        avg_pooling_out = torch.sum(lstm_out, 1) / avg_k

        return avg_pooling_out

        # # self attention
        # self_attention_out = self.attention_layer(lstm_out, lstm_out, lstm_out)
        #
        # # adj attention
        # before_part = lstm_out[:-1]
        # after_part = lstm_out[1:]
        #
        # after_attention_out = self.attention_layer(after_part, before_part, before_part)
        # before_attention_out = self.attention_layer(before_part, after_part, after_part)
        #
        # size_a, size_b = lstm_out.shape[1:]
        # zero_append = torch.zeros(1, size_a, size_b).cuda()
        #
        # after_attention_out = torch.cat([after_attention_out, zero_append])
        # before_attention_out = torch.cat([zero_append, before_attention_out])

        # # output
        # if self.adj_attention_op == "none":
        #     max_pooling_out = torch.max(lstm_out, 1)[0]
        #     return max_pooling_out
        #
        # elif self.adj_attention_op == "self":
        #     max_pooling_out = torch.max(self_attention_out, 1)[0]
        #     return max_pooling_out
        #
        # elif self.adj_attention_op == "before":
        #     attention_cat = torch.cat([before_attention_out, self_attention_out], 2)
        #     max_pooling_out = torch.max(attention_cat, 1)[0]
        #     return max_pooling_out
        #
        # elif self.adj_attention_op == "after":
        #     attention_cat = torch.cat([after_attention_out, self_attention_out], 2)
        #     max_pooling_out = torch.max(attention_cat, 1)[0]
        #     return max_pooling_out
        #
        # elif self.adj_attention_op == "whole":
        #     attention_cat = torch.cat([before_attention_out,
        #                                self_attention_out, after_attention_out], 2)
        #     max_pooling_out = torch.max(attention_cat, 1)[0]
        #     return max_pooling_out

class ParagraphEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix,
                 attention_op="none"):
        super(ParagraphEncoder, self).__init__()

        self.sentence_encoder = SentenceEncoder(embedding_dim=embedding_dim,
                                                hidden_dim=hidden_dim,
                                                vocab_size=vocab_size,
                                                word_vec_matrix=word_vec_matrix,
                                                attention_op=attention_op)

        sentence_encoder_dim = 2*embedding_dim

        if attention_op == "before" or attention_op == "after":
            sentence_encoder_dim *= 2
        elif attention_op == "whole":
            sentence_encoder_dim *= 3

        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder_dim, hidden_size=hidden_dim*2,
                                     bidirectional=True, batch_first=True)

        self.qa_score_linear = nn.Sequential(
            nn.Linear(sentence_encoder_dim * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # self.attention_layer = ScaledDotProductAttentionBatch(model_dim=sentence_encoder_dim)

    def question_answer_score(self, question_tensor, answer_tensor):
        abs_part = torch.abs(question_tensor - answer_tensor)
        multiply_part = question_tensor * answer_tensor
        cat_tensor = torch.cat([question_tensor, answer_tensor, abs_part, multiply_part], 0)
        score = self.qa_score_linear(cat_tensor)
        return score

    def forward(self, paragraph, sentence_length_list):
        """
        :param paragraph: torch tensor; shape=(sent_num, max_sent_length)
        :param sentence_length_list: torch tensor; shape=sent_num

        :var sentence_encoder_out: torch tensor; shape=(sent_num, embedding_dimx2=600)
        :var sentence_lstm_out: torch tensor; shape=(1, sent_num, embedding_dimx4=1200)

        :return: attention_out: torch tensor; shape=(sent_num, embedding_dimx4=1200)
        """
        sentence_encoder_out = self.sentence_encoder(paragraph, sentence_length_list)
        rest = MAX_SENTENCE_NUM_IN_PARAGRAPH % sentence_encoder_out.shape[0]
        sentence_encoder_out = nn.ZeroPad2d((0, 0, 0, rest))(sentence_encoder_out)

        sentence_num = sentence_encoder_out.shape[0]
        similarity_list = [self.question_answer_score(sentence_encoder_out[i], sentence_encoder_out[i+1])
                           for i in range(sentence_num-1)]

        return torch.cat(similarity_list).view(1, -1)

        # sentence_lstm_out, _ = self.sentence_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))
        # return sentence_lstm_out.view(sentence_lstm_out.shape[1], sentence_lstm_out.shape[2])

        # attention_out = self.attention_layer(sentence_lstm_out, sentence_lstm_out, sentence_lstm_out)\
        #     .view(sentence_lstm_out.shape[1], -1)
        #
        # return attention_out

class ParagraphBatchProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout,
                 batch_size, attention_op="none"):
        super(ParagraphBatchProcessor, self).__init__()
        self.batch_size = batch_size
        self.paragraph_encoder = ParagraphEncoder(embedding_dim=embedding_dim,
                                                  hidden_dim=hidden_dim,
                                                  vocab_size=vocab_size,
                                                  word_vec_matrix=word_vec_matrix,
                                                  attention_op=attention_op)

        self.paragraph_vector_dim = MAX_SENTENCE_NUM_IN_PARAGRAPH-1 # embedding_dim*4

        self.classifier = nn.Sequential(
            nn.Linear(self.paragraph_vector_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

    def forward(self, paragraph_batch, sentence_length_list_batch):
        assert self.batch_size == len(paragraph_batch)
        assert self.batch_size == len(sentence_length_list_batch)
        paragraph_encoder_out_batch = [self.paragraph_encoder(para, sent_len_list)
                                       for para, sent_len_list in zip(paragraph_batch, sentence_length_list_batch)]

        # # paragraph_vector_batch = [torch.max(para_enc_out, 0)[0] for para_enc_out in paragraph_encoder_out_batch]
        # paragraph_vector_batch = [para_enc_out[-1] for para_enc_out in paragraph_encoder_out_batch]
        #
        # paragraph_tensor_matrix = paragraph_vector_batch[0].view(1, paragraph_vector_batch[0].shape[0])
        # for i in range(1, self.batch_size):
        #     paragraph_vector = paragraph_vector_batch[i].view(1, paragraph_vector_batch[i].shape[0])
        #     paragraph_tensor_matrix = torch.cat([paragraph_tensor_matrix, paragraph_vector])

        paragraph_tensor_matrix = torch.cat(paragraph_encoder_out_batch, 0)

        score_batch = self.classifier(paragraph_tensor_matrix)

        return score_batch

