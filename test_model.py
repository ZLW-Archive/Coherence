import numpy as np
from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, LSTM, Input, Embedding, Concatenate, Lambda, Dropout, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from scipy import stats

from data_proc.read_proc_data import read_proc_data

DATASET = "test"
TAG = "ori_try_3"
CHECKPOINT = ["ori_try_3"]

print("Now check on dataset: {}".format(DATASET))

MODEL_PATH = ["checkpoint/{}/saved_model_weight.h5".format(cp) for cp in CHECKPOINT]
SAVE_PATH = "checkpoint/{}/{}_result.txt".format(TAG, DATASET)
ERROR_PATH = "checkpoint/{}/{}_error.txt".format(TAG, DATASET)

BATCH_SIZE = 100
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 500
MAX_SENTENCE_NUM_IN_PARAGRAPH = 55

# <editor-fold desc="Preparation">
paragraph_raw_data = {DATASET: read_proc_data(DATASET, para_wise=True, pos_need=True)}
paragraph_list = {DATASET: paragraph_raw_data[DATASET][0]}
paragraph_pos = {DATASET: paragraph_raw_data[DATASET][1]}
paragraph_num = {DATASET: len(paragraph_list[DATASET])}

def get_dataset(tag):
    sequence = [para["paragraph"] for para in paragraph_list[tag]]
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

    pos = paragraph_pos[tag]
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            if pos[i][j] >= MAX_SEQUENCE_LENGTH:
                pos[i][j] = -1
    pos = pad_sequences(pos, maxlen=MAX_SENTENCE_NUM_IN_PARAGRAPH, padding='post', value=-1)

    return sequence, pos

def get_generator(tag, batch_size):
    para_num = paragraph_num[tag]
    while True:
        times = para_num // batch_size
        for i in range(times):
            yield paragraph_dataset[tag][0][i * (batch_size): (i + 1) * batch_size]

paragraph_dataset = {DATASET: get_dataset(DATASET)}
paragraph_loader = {DATASET: get_generator(DATASET, BATCH_SIZE)}

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
# </editor-fold>

embedding_matrix = np.load("data/word_vector_matrix.npy")

vocab_size = embedding_matrix.shape[0]

embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            mask_zero=True,
                            trainable=False)
side_embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 mask_zero=False,
                                 trainable=False)

def SkipFlow(lstm_dim=50, lr=1e-4, lr_decay=1e-6, k=5, eta=3, delta=50, activation="relu", maxlen=MAX_SEQUENCE_LENGTH,
             dropout=0.5, seed=None):
    e = Input(name='essay', shape=(maxlen,))
    embed = embedding_layer(e)
    side_embed = side_embedding_layer(e)
    lstm_layer = LSTM(lstm_dim, return_sequences=True)
    hidden_states = lstm_layer(embed)
    side_hidden_states = lstm_layer(side_embed)
    htm = TemporalMeanPooling()(hidden_states)
    tensor_layer = NeuralTensorLayer(output_dim=k, input_dim=lstm_dim)
    pairs = [((eta + i * delta) % maxlen, (eta + i * delta + delta) % maxlen) for i in range(maxlen // delta)]
    hidden_pairs = [
        (Lambda(lambda t: t[:, p[0], :])(side_hidden_states), Lambda(lambda t: t[:, p[1], :])(side_hidden_states)) for p
        in pairs]
    sigmoid = Dense(1, activation="sigmoid", kernel_initializer=initializers.glorot_normal(seed=seed))
    coherence = [sigmoid(tensor_layer([hp[0], hp[1]])) for hp in hidden_pairs]
    co_tm = Concatenate()(coherence[:] + [htm])

    dense = Dense(256, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(co_tm)
    # dense = Dropout(dropout)(dense)
    dense = Dense(128, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    # dense = Dropout(dropout)(dense)
    dense = Dense(64, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)

    out = Dense(2, activation="sigmoid")(dense)
    mod = Model(inputs=e, outputs=out)
    adam = Adam(lr=lr, decay=lr_decay)
    mod.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return mod

model = SkipFlow(lstm_dim=50, lr=2e-4, lr_decay=2e-6, k=4, eta=13, delta=50, activation="relu", seed=None)

file = open(SAVE_PATH, "w", encoding="utf-8")
error = open(ERROR_PATH, "w", encoding="utf-8")
total_num = len(paragraph_list[DATASET])
error_num = 0
error_list = [[], []]

pred_result_cnt = [0 for _ in range(total_num)]
pred_result = []

for path in MODEL_PATH:
    model.load_weights(path)
    pred = model.predict_generator(paragraph_loader[DATASET], steps=paragraph_num[DATASET] // BATCH_SIZE, verbose=1)

    for i in range(pred.shape[0]):
        score = pred[i]
        if score[0] > score[1]:
            pred_result_cnt[i] -= 1
        else:
            pred_result_cnt[i] += 1

for i in range(total_num):
    p = -1
    if pred_result_cnt[i] > 0:
        file.write("1\n")
        p = 1
    else:
        file.write("0\n")
        p = 0
    pred_result.append(p)

    if p != paragraph_list[DATASET][i]["tag"]:
        error.write("{}\n".format(i))
        error_num += 1
        error_list[paragraph_list[DATASET][i]["tag"]].append(i)

print("Acc: {:.5f}".format((total_num-error_num)/total_num))
print("0-Acc: {:.5f}".format((total_num/2-len(error_list[0]))/(total_num/2)))
print("1-Acc: {:.5f}".format((total_num/2-len(error_list[1]))/(total_num/2)))

print("Test Finish!")
