import numpy as np
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from scipy import stats

from data_proc.read_proc_data import read_proc_data

MODEL_PATH = "model.h5"

BATCH_SIZE = 100
MAX_SEQUENCE_LENGTH = 500
MAX_SENTENCE_NUM_IN_PARAGRAPH = 55

paragraph_raw_data = {"test": read_proc_data("test", para_wise=True, pos_need=True)}
paragraph_list = {"test": paragraph_raw_data["test"][0]}
paragraph_pos = {"test": paragraph_raw_data["test"][1]}
paragraph_num = {"test": len(paragraph_list["test"])}

def get_dataset(tag, shuffle=False):
    sequence = [para["paragraph"] for para in paragraph_list[tag]]
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    pos = paragraph_pos[tag]
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            if pos[i][j] >= MAX_SEQUENCE_LENGTH:
                pos[i][j] = -1
    pos = pad_sequences(pos, maxlen=MAX_SENTENCE_NUM_IN_PARAGRAPH, padding='post', value=-1)

    if shuffle:
        idx = np.arange(paragraph_num[tag])
        np.random.shuffle(idx)
        sequence = sequence[idx]
        pos = pos[idx]

    return sequence, pos

def get_generator(tag, batch_size):
    para_num = paragraph_num[tag]
    while True:
        times = para_num // batch_size
        for i in range(times):
            yield paragraph_dataset[tag][0][i * (batch_size): (i + 1) * batch_size]

paragraph_dataset = {"test": get_dataset("test", shuffle=False)}
paragraph_loader = {"test": get_generator("test", BATCH_SIZE)}

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

model = load_model(MODEL_PATH)

pred = model.predict_generator(paragraph_loader["test"], steps=paragraph_num["test"] // BATCH_SIZE, verbose=1)

print(123)
