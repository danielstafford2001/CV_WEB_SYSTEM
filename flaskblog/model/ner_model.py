from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
import os
import logging
import numpy as np
from tensorflow.keras.models import model_from_json
import nltk
import json
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
from tensorflow.keras.models import load_model


num_words = 22690
num_tags = 25

#These lines of code used to allow memory grow dynamically on GPU.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')


# NerModel class it has methods to prepare model and make predictions.
class NERModel(object):
    #tags, words and paths to model are stored in cfg.json other things stored in .npy
    def __init__(self, cfg):
        self.tags = cfg['tags']
        self.words = cfg['words']
        full_path = os.path.dirname(__file__)
        self.model = self.prepare_model(os.path.join(full_path, cfg['model_path']))

        self.full_tag_names = cfg['full_tag_names']

        self.word2Idx = np.load(os.path.join(full_path, "word2Idx.npy"), allow_pickle=True).item()
        self.char2Idx = np.load(os.path.join(full_path, "char2Idx.npy"), allow_pickle=True).item()
        self.case2Idx = np.load(os.path.join(full_path, "case2Idx.npy"), allow_pickle=True).item()
        self.idx2Label = np.load(os.path.join(full_path, "idx2Label.npy"), allow_pickle=True).item()

    #this functions loads model architecture and model weights.
    def prepare_model(self, model_path):
        loaded_model = load_model(model_path)
        print("model loaded successfully")
        return loaded_model

    #Understanding the case of a word
    def getCasing(self, word, caseLookup):
        casing = 'other'

        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():  # Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():  # All lower case
            casing = 'allLower'
        elif word.isupper():  # All upper case
            casing = 'allUpper'
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'
        return caseLookup[casing]

    #Creating tensor to model
    def createTensor(self, sentence, word2Idx, case2Idx, char2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']

        wordIndices = []
        caseIndices = []
        charIndices = []

        for word, char in sentence:
            word = str(word)

            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx

            charIdx = []
            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN'])

            wordIndices.append(wordIdx)
            caseIndices.append(self.getCasing(word, case2Idx))
            charIndices.append(charIdx)

        return [wordIndices, caseIndices, charIndices]


    def addCharInformation(self, sentence):
        return [[word, list(str(word))] for word in sentence]


    def padding(self,Sentence):
        Sentence[2] = pad_sequences(Sentence[2],52,padding='post')
        return Sentence

    # This function makes nice looking format from model prediction tags and words

    def process_prediction(self, tokens, tags):

        # for each tag we need to get their pos
        pos_tags = [pos for token, pos in pos_tag(tokens)]
        # then convert it to tuples
        conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
        # now we can create tree
        ne_tree = conlltags2tree(conlltags)

        result = []
        # now we adding all words that has different tag than 'O'
        # format is [(word,tag)]
        for subtree in ne_tree:
            if type(subtree) == Tree:
                label = subtree.label()
                string = " ".join([token for token, pos in subtree.leaves()])
                result.append((string, label))

        #creating a dict in format: {key: []}

        tags_list = dict(zip(self.full_tag_names.keys(), [set() for _ in range(len(self.full_tag_names.keys()))]))

        for value, key in result:
            tags_list[key].add(value)

        final_repr = ""
        for key in tags_list.keys():
            final_repr += self.full_tag_names[key] + ": " + str(list(tags_list[key])) + "\n\n"

        return final_repr

    # prediction for a whole text
    def full_predict(self, text):
        # splitlines in text
        paragraphs = text.splitlines()
        paragraphs = [paragraph.strip() for paragraph in paragraphs]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph] # checking for empty lines

        # after this we need to split our paragraphs into sentences
        res = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            for sentence in sentences:
                res.append(sentence)

        words = []
        tags = []
        for sentence in res:
            w,t = self.predict(sentence)
            words.extend(w)
            tags.extend(t)

        return self.process_prediction(words, tags)


    #prediction function for one sentence
    def predict(self,Sentence):
        #tokenizing the words
        Sentence = words = nltk.word_tokenize(Sentence)
        Sentence = self.addCharInformation(Sentence)
        Sentence = self.padding(self.createTensor(Sentence,self.word2Idx,self.case2Idx,self.char2Idx))

        tokens, casing,char = Sentence
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])

        pred = self.model.predict([tokens, casing,char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        tags = [self.idx2Label[i].strip() for i in pred]

        return words, tags







