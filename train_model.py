import argparse
import os
import pickle
import sys

import numpy as np
from keras import backend as K
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, LSTM, Input, Embedding, Concatenate, Lambda, Dropout, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from scipy import stats

from data_proc.read_proc_data import read_proc_data

# Run Option
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=1, choices=[1, 2])
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--file", type=bool, default=False)
parser.add_argument("--directory", type=str, default="tmp")

args = parser.parse_args()
print(args)

VERBOSE = args.verbose
TO_FILE = args.file
DIRECTORY = "checkpoint/{}".format(args.directory)

if TO_FILE:
    try:
        os.makedirs(DIRECTORY)
    except:
        pass
    output_file = open("{}/output.vstxt".format(DIRECTORY), "w", encoding="utf-8")
    sys.stdout = output_file

# parameter
EMBEDDING_DIM = 300
MAX_NB_WORDS = 4000
MAX_SEQUENCE_LENGTH = 500
MAX_SENTENCE_NUM_IN_PARAGRAPH = 55
DELTA = 20

BATCH_SIZE = args.batch_size
EPOCH = 1000


class NeuralTensorLayer(Layer):
    def __init__(self, output_dim, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(NeuralTensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        k = self.output_dim
        d = self.input_dim
        # truncnorm generate continuous random numbers in given range
        W_val = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
        V_val = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))
        self.W = K.variable(W_val)
        self.V = K.variable(V_val)
        self.b = K.zeros((self.input_dim,))
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs, mask=None):
        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim

        feed_forward = K.dot(K.concatenate([e1, e2]), self.V)

        bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]

        for i in range(k)[1:]:
            btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
            bilinear_tensor_products.append(btp)

        result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward)

        return result

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return batch_size, self.output_dim


class TemporalMeanPooling(Layer):  # conversion from (samples,timesteps,features) to (samples,features)
    def __init__(self, **kwargs):
        super(TemporalMeanPooling, self).__init__(**kwargs)
        # masked values in x (number_of_samples,time)
        self.supports_masking = True
        # Specifies number of dimensions to each layer
        self.input_spec = InputSpec(ndim=3)

    def call(self, x, mask=None):
        if mask is None:
            mask = K.mean(K.ones_like(x), axis=-1)

        mask = K.cast(mask, K.floatx())
        # dimension size single vec/number of samples
        return K.sum(x, axis=-2) / K.sum(mask, axis=-1, keepdims=True)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class ReshapeLayer(Layer):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def call(self, inputs, **kwargs):
        output = K.reshape(inputs, self.shape)
        return output

    def compute_output_shape(self, input_shape):
        return self.shape

    # sp_sent = K.stack([K.stack([hidden_states[i, pos_re[i, j], :] for j in range(max_sent_num)])
    #                    for i in range(batch_size)])


class SpecialStackLayer(Layer):
    def __init__(self, batch_size, max_sent_num, tensor_dim):
        super(SpecialStackLayer, self).__init__()
        self.batch_size = batch_size
        self.max_sent_num = max_sent_num
        self.tensor_dim = tensor_dim

    def call(self, inputs, **kwargs):
        hidden_states = inputs[0]
        pos = inputs[1]
        output = K.stack([K.stack([hidden_states[i, pos[i, j], :] for j in range(self.max_sent_num)])
                          for i in range(self.batch_size)])
        return output

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.max_sent_num, self.tensor_dim


paragraph_raw_data = {
    "train": read_proc_data("train", para_wise=True, pos_need=True),
    "valid": read_proc_data("valid", para_wise=True, pos_need=True),
    # "test": read_proc_data("test")
}

paragraph_list = {
    "train": paragraph_raw_data["train"][0],
    "valid": paragraph_raw_data["valid"][0]
}
paragraph_pos = {
    "train": paragraph_raw_data["train"][1],
    "valid": paragraph_raw_data["valid"][1]
}

paragraph_num = {
    "train": len(paragraph_list["train"]),
    "valid": len(paragraph_list["valid"]),
}


def get_dataset(tag, shuffle=False):
    sequence = [para["paragraph"] for para in paragraph_list[tag]]
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    tag_list = [para["tag"] for para in paragraph_list[tag]]
    tag_list = np.asarray(tag_list)
    tag_list = np_utils.to_categorical(tag_list)

    pos = paragraph_pos[tag]
    # sent_num = [len(p) for p in pos]
    # sent_num = np.asarray(sent_num, dtype=np.int)
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            if pos[i][j] >= MAX_SEQUENCE_LENGTH:
                pos[i][j] = -1
    pos = pad_sequences(pos, maxlen=MAX_SENTENCE_NUM_IN_PARAGRAPH, padding='post', value=-1)
    # pos_mat_total = []
    # for p in pos:
    #     cur = np.zeros((MAX_SENTENCE_NUM_IN_PARAGRAPH, MAX_SEQUENCE_LENGTH), dtype=np.int)
    #     for i in range(len(p)):
    #         if p[i] < MAX_SEQUENCE_LENGTH:
    #             cur[i][p[i]] = 1
    #     pos_mat_total.append(cur)
    # pos_np = np.stack(pos_mat_total, axis=0)

    if shuffle:
        idx = np.arange(paragraph_num[tag])
        np.random.shuffle(idx)
        sequence = sequence[idx]
        tag_list = tag_list[idx]
        pos = pos[idx]
        # sent_num = sent_num[idx]
        # pos_np = pos_np[idx]

    return sequence, pos, tag_list


def get_generator(tag, batch_size):
    para_num = paragraph_num[tag]
    while True:
        times = para_num // batch_size
        for i in range(times):
            # yield [paragraph_dataset[tag][0][i * (batch_size): (i + 1) * batch_size],
            #        paragraph_dataset[tag][1][i * (batch_size): (i + 1) * batch_size]], \
            #       paragraph_dataset[tag][2][i * (batch_size): (i + 1) * batch_size]
            yield paragraph_dataset[tag][0][i * (batch_size): (i + 1) * batch_size], \
                  paragraph_dataset[tag][2][i * (batch_size): (i + 1) * batch_size]


paragraph_dataset = {
    "train": get_dataset("train", shuffle=True),
    "valid": get_dataset("valid", shuffle=True),
    # "test": get_dataset("test")
}

paragraph_loader = {
    "train": get_generator("train", BATCH_SIZE),
    "valid": get_generator("valid", BATCH_SIZE)
}

embedding_matrix = np.load("data/word_vector_matrix.npy")

vocab_size = embedding_matrix.shape[0]

embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            mask_zero=False,
                            trainable=False)
side_embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 mask_zero=False,
                                 trainable=False)


# third_embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
#                                   input_length=MAX_SEQUENCE_LENGTH,
#                                   mask_zero=False,
#                                   trainable=False)


def SkipFlow(lstm_dim=50, lr=1e-4, lr_decay=1e-6, k=5, eta=3, delta=50, activation="relu",
             batch_size=BATCH_SIZE, max_len=MAX_SEQUENCE_LENGTH, max_sent_num=MAX_SENTENCE_NUM_IN_PARAGRAPH,
             seed=None):
    print("Start to Build Model ...")
    e = Input(name="essay", shape=(max_len,))
    # pos = Input(name="pos", shape=(max_sent_num, ), dtype="int32")

    embed = embedding_layer(e)
    side_embed = side_embedding_layer(e)
    # third_embed = third_embedding_layer(e)

    lstm_layer = Bidirectional(LSTM(lstm_dim, activation="relu", return_sequences=True))

    hidden_states = ReshapeLayer((batch_size, max_len, 2 * lstm_dim))(lstm_layer(embed))
    side_hidden_states = lstm_layer(side_embed)
    # third_states = ReshapeLayer((batch_size, max_len, 2*lstm_dim))(lstm_layer(third_embed))

    # sent_lstm_layer = LSTM(lstm_dim, return_sequences=False)

    # pos_re = ReshapeLayer((batch_size, max_sent_num))(pos)
    # sp_sent = SpecialStackLayer(batch_size, max_sent_num, 2*lstm_dim)([hidden_states, pos_re])
    # # sp_sent_lstm = sent_lstm_layer(sp_sent)
    # sp_sent_mp = TemporalMeanPooling()(sp_sent)

    htm = TemporalMeanPooling()(hidden_states)
    tensor_layer = NeuralTensorLayer(output_dim=k, input_dim=2 * lstm_dim)
    pairs = [((eta + i * delta) % max_len, (eta + i * delta + delta) % max_len) for i in range(max_len // delta)]
    hidden_pairs = [
        (Lambda(lambda t: t[:, p[0], :])(side_hidden_states),
         Lambda(lambda t: t[:, p[1], :])(side_hidden_states))
        for p in pairs]
    sigmoid_dense = Dense(1, activation="sigmoid", kernel_initializer=initializers.glorot_normal(seed=seed))
    coherence = [ReshapeLayer((batch_size, 1))(sigmoid_dense(tensor_layer([hp[0], hp[1]]))) for hp in hidden_pairs]
    # co_tm = Concatenate()(coherence + [htm] + [sp_sent_lstm])
    co_tm = Concatenate()(coherence + [htm])
    # co_tm = Concatenate()(coherence + [sp_sent_mp])

    dense = Dense(256, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(co_tm)
    dense = Dropout(0.5, seed=seed)(dense)
    dense = Dense(128, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    dense = Dropout(0.5, seed=seed)(dense)
    dense = Dense(64, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)

    out = Dense(2, activation="sigmoid")(dense)
    mod = Model(inputs=e, outputs=out)
    # mod = Model(inputs=[e, pos], outputs=[out])
    adam = Adam(lr=lr, decay=lr_decay)
    mod.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print("Finish Building Model ...")

    return mod


earlystopping = EarlyStopping(monitor='val_loss', patience=5)
model = SkipFlow(lstm_dim=50, lr=2e-4, lr_decay=2e-6, k=4, eta=13, delta=50, activation="relu", seed=None)

train_log = model.fit_generator(paragraph_loader["train"], steps_per_epoch=paragraph_num["train"] // BATCH_SIZE,
                                epochs=EPOCH, verbose=VERBOSE, callbacks=[earlystopping],
                                validation_data=paragraph_loader["valid"],
                                validation_steps=paragraph_num["valid"] // BATCH_SIZE)

model.save_weights("{}/saved_model_weight.h5".format(DIRECTORY))

with open("{}/trainHistory".format(DIRECTORY), "wb") as file:
    pickle.dump(train_log.history, file)
