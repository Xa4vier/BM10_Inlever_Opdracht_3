'''
 -*- coding: utf-8 -*-

Xavier van Egdom
Hogeschool Zuyd HBO-ICT

Creating Model and Tokenizer
'''
from __future__ import print_function
import os, os.path
import numpy as np
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from random import shuffle

from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD

import _pickle as cPickle
import pickle
import sys

# save the model
def save_model(model, name):
    #open/create file
    with open(name + '.pkl', 'wb') as fid:
        # save file
        cPickle.dump(model, fid)
        print(name + " tokenizer successfully saved to disk")

# tokenizer saving
def save_tokenizer(tokenizer, name):
    #open/create file
    with open(name + '_tokenizer.pickle', 'wb') as handle:
        # save file
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(name + " model successfully created and saved to disk!")
        
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='latin-1')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # making all tokens lowercase
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('dutch'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

def create_tokenizer(train_docs):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    return tokenizer

# prepare bag of words encoding of docs
def prepare_data(train_docs, mode, tokenizer):
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	return Xtrain, tokenizer

# prepare bag of words encoding of train docs
def prepare_data_chunks(train_docs, y_train, mode, tokenizer, chunk_size):
    # encode training data set
    for i in range(0, len(train_docs), chunk_size):
        train_docs_chunks = train_docs[i:i + chunk_size]
        Xtrain_chunks = tokenizer.texts_to_matrix(train_docs_chunks, mode=mode)
        
        y_train_chunks = y_train[i:i + chunk_size]

        yield Xtrain_chunks, y_train_chunks

# get all folders, all text files in the folders, and the item count of the folders
def get_all_folders(allowed_folders = False):
    # load all training reviews
    folders = [x[0] for x in os.walk("text/")]
    folders.pop(0)
    
    if allowed_folders: folders = [folder for folder in folders if folder in allowed_folders]
        
    X = [] # contain all the text files
    lenght = [] #the lenght is the item count in the maps of folders
    i = 0
    
    while i < len(folders): 
        X += process_docs(folders[i], vocab)
        if i == 0: 
            lenght.append(len(X))
        else: 
            lenght.append(len(X) - sum(lenght))
        i+=1
        
    return X, folders, lenght

# shuffle the sets
def shuffle_sets(X, y):    
    list_to_shuffle = []
    #join the list
    for i in range(len(X)):
        temp = [X[i], y[i]]
        list_to_shuffle.append(temp)
    
    shuffle(list_to_shuffle)
    #disjoin the list
    X = []
    y = []
    for item in list_to_shuffle:
        X.append(item[0])
        y.append(item[1])
    return X, y 

# make labels for more then one classification
def make_labels_all(folders, lenghts):
    y = []
    label = int()
    for i in range(len(folders)): 
        if folders[i] in ["text/NL"]:
            label = 0
            
        if folders[i] in ["text/ES"]:
            label = 1
                    
        y += [label for _ in range(lenghts[i])]
    return y 

# evalution of the model, use chunk_size_training to set the chunk_size for beter memory use
def create_model(train_docs, y_train,  mode, model, name):
    # fit a logistic regression model to the data
    tokenizer = create_tokenizer(train_docs)
    # create generatior for training data 
    X_train, tokenizer = prepare_data(train_docs, mode, tokenizer)
    # train the model by partial fit methode
    model.fit(X_train, y_train)
    # save tokenizer
    save_tokenizer(tokenizer, name)
    # save model
    save_model(model, name)

# load the vocabulary
vocab_filename = 'vocab_trans.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# get all files, folders and item count in folders
train_docs, folders, lenght = get_all_folders()#allowed_folder)
# make labels
y_train = make_labels_all(folders, lenght)
# Shuffle the whole set
train_docs, y_train = shuffle_sets(train_docs, y_train)
# Create model
create_model(train_docs, y_train, 'tfidf', SGDClassifier(loss = 'log'), 'model_tfidf_LR_sport_loss_log')