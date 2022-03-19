import tensorflow as tf
from model import text_preprocessing
from random import choice
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


text = """

Daniel Stafford
•	Completed a full stack trade processing system using Python. This required both Bloomberg and Linedata LGH (Portfolio Management System). The project enabled traders to input their trade information for automatic uploading, saving time and increasing the flow of data.
•	Completed daily reports on P&L, where my understanding of trading strategies was tested each day. For example, the fund had multiple option trades, so I would use payoff diagrams and other tools to predict potential reasons for them. My understanding of traders’ adjustments enabled me to provide daily portfolio summaries to investors, which improved their understanding of the funds economic outlook.
•	Created a Python script to retrieve portfolio quantities from the backend to inform investors of portfolio holdings. This resulted in closer liaison between investors and the fund.
•	Following this internship, Inverewe employed me on a freelance basis, to complete multiple quantitative and qualitative projects for the future.

"""

new_model = tf.keras.models.load_model('my_model.h5')

x_train, x_test, y_train, y_test, words, tags, max_len,X,y, num_words,num_tags,word2idx, tag2idx = text_preprocessing()

#my_sentence = []
for sentence in sent_tokenize(text):
  # print(sentence, '\n\n')
   for word in word_tokenize(sentence):
      print(word, '\n\n')
      

  #print('ITERATION')
  #if word in word2idx:
  #  my_sentence.append(word2idx[word])
  #else:
  #  my_sentence.append(choice(list(set([x for x in range(0, 100000)]) - set(range(0,len(word2idx))))))

#padded_input =pad_sequences(maxlen=max_len, sequences=[my_sentence], padding="post", value=num_words-1)

#print(padded_input[0])