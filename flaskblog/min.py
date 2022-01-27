import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split

data = pd.read_csv('1-3000.csv')
data = data.fillna(method="ffill")

class SentenceGetter(object):
    def __init__(self, data):#data= dataset
        self.n_sent = 1#number of sentences 
        self.data = data

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                            s["POS"].values.tolist(),
                                                            s["Tag"].values.tolist())]# group into tuples(Word, POS, tag)
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)# grouping by sentence and then apply agg_func to each sentence
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data) #getting data into tuples
sentences = getter.sentences

words = list(set(data["Word"].values))
num_words = len(words)

tags = list(set(data["Tag"].values))
num_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

max_len = 60

X = [[word2idx[w[0]] for w in s] for s in sentences] 
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y= [to_categorical(i, num_classes = num_tags) for i in y]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()

model.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])