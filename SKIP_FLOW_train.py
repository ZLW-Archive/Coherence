import random

import keras.layers  as  klayers
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, Embedding, GlobalAveragePooling1D, Concatenate, Activation, Lambda, \
    BatchNormalization, Convolution1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers
from keras.utils import np_utils
from scipy import stats
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import Word2Vec, doc2vec
import nltk
from sklearn.metrics import cohen_kappa_score

from data_proc.read_proc_data import read_proc_data

# parameter
EMBEDDING_DIM = 300
MAX_NB_WORDS = 4000
MAX_SEQUENCE_LENGTH = 500
VALIDATION_SPLIT = 0.20
DELTA = 20


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
        return (batch_size, self.output_dim)


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

    def compute_mask(self, input, mask):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


paragraph_list = {
    "train": read_proc_data("train", para_wise=True, sample=10000),
    "valid": read_proc_data("valid", para_wise=True, sample=100),
    # "test": read_proc_data("test")
}


def get_dataset(tag, shuffle=False):
    sequence = [para["paragraph"] for para in paragraph_list[tag]]
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, truncating="post")
    tag_list = [para["tag"] for para in paragraph_list[tag]]
    tag_list = np.asarray(tag_list)
    tag_list = np_utils.to_categorical(tag_list)

    if shuffle:
        idx = np.arange(len(paragraph_list[tag]))
        np.random.shuffle(idx)
        sequence = sequence[idx]
        tag_list = tag_list[idx]

    return sequence, tag_list


paragraph_dataset = {
    "train": get_dataset("train", shuffle=True),
    "valid": get_dataset("valid"),
    # "test": get_dataset("test")
}

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


def SKIPFLOW(lstm_dim=50, lr=1e-4, lr_decay=1e-6, k=5, eta=3, delta=50, activation="relu", maxlen=MAX_SEQUENCE_LENGTH,
             seed=None):
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
    dense = Dense(128, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    dense = Dense(64, activation=activation, kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    out = Dense(2, activation="sigmoid")(dense)
    model = Model(inputs=e, outputs=out)
    adam = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


earlystopping = EarlyStopping(monitor='val_loss', patience=10)
sf_1 = SKIPFLOW(lstm_dim=50, lr=2e-4, lr_decay=2e-6, k=4, eta=13, delta=50, activation="relu", seed=None)

epochs = 1000
hist = sf_1.fit(paragraph_dataset["train"][0], paragraph_dataset["train"][1], batch_size=1000, epochs=epochs,
                validation_data=(paragraph_dataset["valid"][0], paragraph_dataset["valid"][1]), callbacks=[earlystopping])

