import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.Attention import ScaledDotProductAttentionBatch

class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec_matrix,
                 adj_attention_op="self"):
        super(SentenceEncoder, self).__init__()
        self.adj_attention_op = adj_attention_op

        # word level
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(word_vec_matrix))

        self.bilstm_layer = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    bidirectional=True,
                                    batch_first=True)

        self.attention_layer = ScaledDotProductAttentionBatch(model_dim=2*embedding_dim)

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
        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # unsort
        lstm_out = lstm_out[desorted_indices]

        # self attention
        self_attention_out = self.attention_layer(lstm_out, lstm_out, lstm_out)\

        # adj attention
        before_part = lstm_out[:-1]
        after_part = lstm_out[1:]

        after_attention_out = self.attention_layer(after_part, before_part, before_part)
        before_attention_out = self.attention_layer(before_part, after_part, after_part)

        size_a, size_b = lstm_out.shape[1:]
        zero_append = torch.zeros(1, size_a, size_b)

        after_attention_out = torch.cat([after_attention_out, zero_append])
        before_attention_out = torch.cat([zero_append, before_attention_out])

        # output
        if self.adj_attention_op == "self":
            max_pooling_out = torch.max(self_attention_out, 1)[0]
            return max_pooling_out

        elif self.adj_attention_op == "before":
            attention_cat = torch.cat([before_attention_out, self_attention_out], 2)
            max_pooling_out = torch.max(attention_cat, 1)[0]
            return max_pooling_out

        elif self.adj_attention_op == "after":
            attention_cat = torch.cat([after_attention_out, self_attention_out], 2)
            max_pooling_out = torch.max(attention_cat, 1)[0]
            return max_pooling_out

        elif self.adj_attention_op == "whole":
            attention_cat = torch.cat([before_attention_out,
                                       self_attention_out, after_attention_out], 2)
            max_pooling_out = torch.max(attention_cat, 1)[0]
            return max_pooling_out

class ParagraphEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, vocab_size, tagset_size, word_vec_matrix, dropout,
                 adj_attention_op="self"):
        super(ParagraphEncoder, self).__init__()

        self.sentence_encoder = SentenceEncoder(embedding_dim=embedding_dim,
                                                hidden_dim=hidden_dim,
                                                vocab_size=vocab_size,
                                                word_vec_matrix=word_vec_matrix,
                                                adj_attention_op=adj_attention_op)

        sentence_encoder_dim = 2*embedding_dim

        if adj_attention_op == "before" or adj_attention_op == "after":
            sentence_encoder_dim *= 2
        elif adj_attention_op == "whole":
            sentence_encoder_dim *= 3

        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder_dim, hidden_size=hidden_dim*2,
                                     bidirectional=True, batch_first=True)

        self.attention_layer = ScaledDotProductAttentionBatch(model_dim=sentence_encoder_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim * 2, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tagset_size)
        )

        self.qa_score_linear = nn.Sequential(
            nn.Linear(sentence_encoder_dim*4, 100),
            nn.Linear(100, 1)
        )

    def forward(self, paragraph_list):
        sentence_encoder_out = self.sentence_encoder(sentence_list, sentence_length_list)
        sentence_lstm_out, _ = self.sentence_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))

        # attention_out = self.attention_layer(sent_lstm_out, sent_lstm_out, sent_lstm_out)
        # tag_space = self.classifier(attention_out.view(attention_out.shape[1], -1))

        tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))

        # tag_space = self.classifier(sentence_encoder_out)

        return sentence_encoder_out, tag_space

    def question_answer_score(self, question_tensor, answer_tensor):
        abs_part = torch.abs(question_tensor - answer_tensor)
        multiply_part = question_tensor * answer_tensor
        cat_tensor = torch.cat([question_tensor, answer_tensor, abs_part, multiply_part], 0)
        score = self.qa_score_linear(cat_tensor)
        return score
        # question_matrix = question_tensor.view(1, -1)
        # answer_matrix = answer_tensor.view(1, -1)
        #
        # return torch.mm(question_matrix, answer_matrix.t()).view([])

    # multitask loss
    def get_loss(self, sentence_encoder_out, tag_space, emotion_loss_func, targets):
        # sentence_encoder_out = self.sentence_encoder(sentence_tuple)
        # sent_lstm_out, _ = self.sent_lstm(sentence_encoder_out.view(1, sentence_encoder_out.shape[0], -1))
        #
        # # attention_out = self.attention_layer(sent_lstm_out, sent_lstm_out, sent_lstm_out)
        # # tag_space = self.classifier(attention_out.view(attention_out.shape[1], -1))
        #
        # tag_space = self.classifier(sent_lstm_out.view(sent_lstm_out.shape[1], -1))
        #
        # # tag_space = self.classifier(sentence_encoder_out)

        emotion_loss = emotion_loss_func(tag_space, targets)

        # return emotion_loss, emotion_loss, 0

        # self.loss = tf.reduce_mean(tf.nn.relu(1 + self.qa_score_1 - self.qa_score_2)
        sentence_num = len(sentence_encoder_out)

        batch_loss = []
        for i in range(sentence_num - 1):
            sent_loss = []

            sent = sentence_encoder_out[i]
            next_sent = sentence_encoder_out[i+1]
            true_score = self.question_answer_score(sent, next_sent)

            # rand_sample = random.sample([i for i in range(sentence_num)], 10)

            for j in range(sentence_num):
                if j != i and j != i+1:
                    false_sent = sentence_encoder_out[j]
                    false_score = self.question_answer_score(sent, false_sent)
                    sent_loss.append(F.relu(1 - true_score + false_score))

            sent_loss_sum = sum(sent_loss)
            sent_loss_mean = sent_loss_sum / len(sent_loss)
            batch_loss.append(sent_loss_mean)

        batch_loss_sum = sum(batch_loss)
        batch_loss_mean = batch_loss_sum / len(batch_loss)

        loss = emotion_loss + batch_loss_mean
        # print("emotion loss: {:.3f} answer loss: {:.3f}".format(float(emotion_loss), float(batch_loss_mean)))

        # loss = batch_loss_mean
        # print("answer loss: {:.6f}".format(float(loss)))

        return loss, emotion_loss, batch_loss_mean


