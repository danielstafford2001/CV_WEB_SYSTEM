import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback
#from config import Config
#from data_utils import CoNLLDataset
import os
from tensorflow import keras
np.random.seed(0)
plt.style.use("ggplot")
print(tf.version.VERSION)
import time
import sys
import logging
import numpy as np

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

class Config():
    def __init__(self, load=True):
        self.logger = get_logger(self.path_log)
    dir_output = "results/test/"
    path_log = dir_output + "log.txt"

class CoNLLDataset(object):
    def __init__(self, filename):
        self.filename = filename
    def get_head(filename):
        data = pd.read_csv(filename)
        data = data.fillna(method="ffill")
        return data

#dev = CoNLLDataset.get_head('/Users/danielstafford/Desktop/getting_dataset_minimum_from_csv/data/1_8600.csv')
#print(dev)
#print(dev.head())


data = CoNLLDataset.get_head(
    '/Users/danielstafford/Desktop/3rd year/Project/CV Work/datasets/Labbeled/newResume/1-8600.csv')
num_words = 22690
num_tags = 25
tags = list(set(data["Tag"].values))
words = list(set(data["Word"].values))
max_len = 150
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

class NERModel(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
    def get_sentences(self):
        self.logger.info("Getting sentences")
        global sentences
        getter = SentenceGetter(data)
        sentences = getter.sentences
        return sentences
    def get_tag_index(self):
        self.logger.info("Getting tags index")
        global tag2idx
        tag2idx = {t: i for i, t in enumerate(tags)}
        return tag2idx
    def get_word_index(self):
        self.logger.info("Getting words index")
        global word2idx
        word2idx = {w: i + 1 for i, w in enumerate(words)}
        return word2idx
    def padding_sequences(self):
        self.logger.info("Padding sequence")
        global X,y
        X = [[word2idx[w[0]] for w in s] for s in sentences]
        X = pad_sequences(maxlen=max_len, sequences=X,
                  padding="post", value=num_words-1)
        y = [[tag2idx[w[2]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=max_len, sequences=y,
                  padding="post", value=tag2idx["O"])
        y = [to_categorical(i, num_classes=num_tags) for i in y]
    def define_model(self):
        self.logger.info("Defining the model")
        global x_train, x_test, y_train, y_test, X, y, input_word, model, out

        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1)
        input_word = Input(shape=(max_len,))
        model = Embedding(input_dim=num_words, output_dim=max_len,
                  input_length=max_len)(input_word)
        model = SpatialDropout1D(0.1)(model)
        model = Bidirectional(
        LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
        model = Model(input_word, out)
    def compile_model(self):
        self.logger.info("Compiling the model")
        global model
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.save('path_to_my_model.h5')
        return model
    def build_model(self):
        self.logger.info("Building the model")
        global history, model
        history = (model.fit(
            x_train, np.array(y_train),
            validation_split=0.2,
            batch_size=32,
            epochs=1,
            verbose=1,
        ))
    def evaluate_model(self):
        self.logger.info("Evaluating the model")
        global model
        model.evaluate(x_test, np.array(y_test))
    def build(self):
        self.get_sentences()
        self.get_tag_index()
        self.get_word_index()
        #self.padding_sequences()
        #self.define_model()
        #self.compile_model()
        #self.build_model()
        #self.evaluate_model()

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        def agg_func(s): return [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["POS"].values.tolist(),
                            s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
def main():
    config = Config()
    model = NERModel(config)
    model.build()
    print("-------------------")
    new_model = tf.keras.models.load_model('path_to_my_model.h5')
    new_model.summary()
    new_model.evaluate(x_test, np.array(y_test))
main()