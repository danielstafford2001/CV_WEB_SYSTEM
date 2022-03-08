# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback
import matplotlib.pyplot as plt
from random import choice
import nltk
from nltk.tokenize import word_tokenize

np.random.seed(0)
plt.style.use("ggplot")

def text_preprocessing():
    data = pd.read_csv('1-3000.csv')
    data = data.fillna(method="ffill")#fill any empty rows with the row above

   # print("Unique words in corpus:", data['Word'].nunique())
   # print("Unique tags in corpus:", data['Tag'].nunique())

  #  print(data['Tag'].unique())

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

    return x_train, x_test, y_train, y_test, words, tags, max_len,X,y, num_words,num_tags, word2idx, tag2idx

#text_preprocessing()

def create_model():
    x_train, x_test, y_train, y_test, words, tags, max_len,X,y, num_words,num_tags =text_preprocessing()
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

    return model,X,y,x_train, x_test,y_train, y_test

#create_model()

def saving_model():
    model,X,y,x_train,x_test,y_train, y_test= create_model()
    model.fit(x_train, np.array(y_train),validation_split=0.2,batch_size=32, epochs=10,verbose=1)
    model.save('my_model.h5')
    return model

#saving_model()

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

x_train, x_test, y_train, y_test, words, tags, max_len,X,y, num_words,num_tags,word2idx, tag2idx = text_preprocessing()

nice_input = 'Python'

from random import choice
import nltk
from nltk.tokenize import word_tokenize

my_sentence = []
for word in word_tokenize(nice_input):
  print('ITERATION')
  if word in word2idx:
    my_sentence.append(word2idx[word])
  else:
    my_sentence.append(choice(list(set([x for x in range(0, 100000)]) - set(range(0,len(word2idx))))))

padded_input =pad_sequences(maxlen=max_len, sequences=[my_sentence], padding="post", value=num_words-1)

#print(padded_input[0])

#for entry in padded_input[0]:
#  key = [k for k, v in word2idx.items() if v == entry]
#  print(key)

print('HERE')
p = new_model.predict(np.array(padded_input))
p = np.argmax(p, axis=-1)
for pred in p[0]:
    print(tags[pred])

#loss, acc = new_model.evaluate(x_test, y_test)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#i = np.random.randint(0, x_test.shape[0]) 
#p = new_model.predict(np.array([x_test[i]]))
#p = np.argmax(p, axis=-1)
#y_true = np.argmax(np.array(y_test), axis=-1)[i]
#print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
#print("-" *30)
#for w, true, pred in zip(x_test[i], y_true, p[0]):
#    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

