'''
 -*- coding: utf-8 -*-

Xavier van Egdom
Hogeschool Zuyd HBO-ICT

Evaluate Models
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

from sklearn.metrics import confusion_matrix
import xlsxwriter
import sys

# create excelsheet
def create_excel(name, data):
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet()
    
    row = 0
    
    for col, data in enumerate(data):
        worksheet.write_column(row, col, data)
        
    workbook.close()

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
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest

# generator to prepare bag of words encoding of docs
def prepare_test_data(test_doc, mode, tokenizer, chunk_size):
    # loop trough test_doc in steps of chunk_size
    for i in range(0, len(test_doc), chunk_size):
        # slice test_doc in chunks
        test_docs_chunks = test_docs[i:i + chunk_size]
        # encode training data set
        Xtest_chunks = tokenizer.texts_to_matrix(test_docs_chunks, mode=mode)
        yield Xtest_chunks

# generator to prepare bag of words encoding of docs
def prepare_train_data(train_docs, y_train, mode, tokenizer, chunk_size):
    
    for i in range(0, len(train_docs), chunk_size):
        # slice train_doc in chunks
        train_docs_chunks = train_docs[i:i + chunk_size]
        # encode training data set
        Xtrain_chunks = tokenizer.texts_to_matrix(train_docs_chunks, mode=mode) 
        # slice y_train in chunks
        y_train_chunks = y_train[i:i + chunk_size]
        yield Xtrain_chunks, y_train_chunks

# get all folders, all text files in the folders, and the item count of the folders
def get_all_folders(allowed_folders = False):
    # load all training reviews
    folders = [x[0] for x in os.walk("text/")]
    # delete first item, its a mac file
    folders.pop(0) 
    
    # filter folders that are allowed
    if allowed_folders: folders = [folder for folder in folders if folder in allowed_folders]      
    
    X = [] # contain all the text files
    lenght = [] #the lenght is the item count in the maps of folders    
    for i in range(len(folders)): 
        X += process_docs(folders[i], vocab)
        if i == 0: 
            lenght.append(len(X))
        else: 
            lenght.append(len(X) - sum(lenght))
        
    return X, folders, lenght

# shuffle the sets
def shuffle_sets(X, y):    
    list_to_shuffle = []
    #join the list
    for i in range(len(X)):
        temp = (X[i],y[i])
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
def make_labels_groups(folders, lenghts):
    y = []
    label = int()
    for i in range(len(folders)): 
        if folders[i] in ["text/binnenland", "text/buitenland", "text/politiek"]:
            label = 0
            
        elif folders[i] in ["text/geldzaken", "text/beurs", "text/carriere"]: 
            label = 1
                        
        elif folders[i] in ["text/voetbal", "text/champions-league", "text/knvb-beker",
                            "text/formule-1", "text/schaatsen", "text/sport-overig"]: 
            label = 2
            
        elif folders[i] in ["text/achterklap", "text/film", "text/muziek", "text/media", "text/cultuur-overig"]:
        
            label = 3
            
        elif folders[i] in ["text/internet", "text/gadgets", "text/games", "text/mobiel"]:
            
            label = 4
            
        elif folders[i] in ["text/gezodheid", "text/eten-en-drinken", "text/reizen"]:
            
            label = 5
            
        elif folders[i] in ["text/dieren"]:
            
            label = 6
        
        elif folders[i] in ["text/wetenschap"]:
            
            label = 7
        
        y += [label for _ in range(lenghts[i])]
    return y

# make labels for more then one classification
def make_labels_all(folders, lenghts):
    y = []
    label = int()
    for i in range(len(folders)): 
        if folders[i] in ["text/NL"]:
            label = 0
            
        if folders[i] in ["text/ES"]:
            label = 1
            
        if folders[i] in ["text/politiek"]:
            label = 2
            
        elif folders[i] in ["text/geldzaken"]: 
            label = 3
            
        elif folders[i] in ["text/beurs"]: 
            label = 4
            
        elif folders[i] in ["text/carriere"]: 
            label = 5
                        
        elif folders[i] in ["text/voetbal", "text/champions-league", "text/knvb-beker"]: 
            label = 6
        
        y += [label for _ in range(lenghts[i])]
    return y

 
#make training set and test set
def train_test_split(X, y, training_size):
    # slice training and test X set
    X_train = X[:int(len(X) * training_size)]
    X_test = X[int(len(X) * training_size):]
    # slice training and test y set
    y_train = y[:int(len(y) * training_size)]
    y_test = y[int(len(y) * training_size):]
       
    return X_train, X_test, y_train, y_test

# evalution of the model, use chunk_size_training to set the chunk_size for beter memory use
def evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, mode, classifier, chunk_size_training, name):
    # fit a logistic regression model to the data
    tokenizer = create_tokenizer(train_docs)
    # create generatior for training data 
    train_data = prepare_train_data(train_docs, ytrain, mode, tokenizer, chunk_size_training)    
    # Train model
    i = 0
    # loop through the generator train_data
    for X_chunk, y_chunk in train_data:
        # train the model by partial fit methode
        classifier.partial_fit(X_chunk, y_chunk, classes=np.unique(ytrain))
        i += chunk_size_training
        print('\r{0:d}'.format(i), end='')
        sys.stdout.flush()
     
    # create generatior for test data  
    test_data = prepare_test_data(test_docs, mode, tokenizer, chunk_size=1)
    
    # create int64 array to store test results in
    ypred = np.array((1,))
    # loop through test_data generator
    for X_chunk in test_data:
    # Now make predictions with trained model
        ypred = np.append(ypred, classifier.predict(X_chunk))
    # delete the first item of the array that was created when the array was instantiated
    ypred = np.delete(ypred, 0, 0)
    #save model in Excel file
    create_excel('cm/' + name, confusion_matrix(ytest, ypred))
    
    print("\n" + name + " evaluated successfully and saved to disk!")

def evaluation_model(train_docs, test_docs, ytrain, ytest, mode, classifier, name):   
    # prepare X_train and X_test
    Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
    # fit training data on model
    classifier.fit(Xtrain, ytrain)
    # test model
    ypred = classifier.predict(Xtest)
    
    cm = confusion_matrix(ytest, ypred)
    #save model in Excel file
    create_excel('cm/' + name, cm)
    print("\n" + name + " evaluated successfully and saved to disk!")
    return cm
    

# load the vocabulary
vocab_filename = 'vocab_trans.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

#allowed_folders = ["text/binnenland", "text/buitenland", "text/politiek"]

# get all files, folders and item count in folders
X, folders, lenght = get_all_folders()#allowed_folders)
lenght[0] = lenght[0] - 1  
del X[0]
# make labels
y = make_labels_all(folders, lenght)
# Shuffle the whole set
X, y = shuffle_sets(X,y)
#make training set and test set
train_docs, test_docs, ytrain, ytest = train_test_split(X, y, 0.8)

# REMINDER
modes = ['binary', 'count', 'tfidf', 'freq']
classifiers = [SGDClassifier(), GaussianNB(), SVC(), KNeighborsClassifier()]

#cm = confusion matrix
cm = evaluation_model(train_docs, test_docs, ytrain, ytest, 'binary', SGDClassifier(loss = 'log'), 'cm_lr_b_loss_log_group')
cm = evaluation_model(train_docs, test_docs, ytrain, ytest, 'count', SGDClassifier(loss = 'log'), 'cm_lr_c_loss_log_group')
cm = evaluation_model(train_docs, test_docs, ytrain, ytest, 'tfidf', SGDClassifier(loss = 'log'), 'cm_lr_t_loss_log_group')
cm = evaluation_model(train_docs, test_docs, ytrain, ytest, 'freq', SGDClassifier(loss = 'log'), 'cm_lr_f_loss_log_group')

evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'binary', GaussianNB(), 2000, 'cm_nb_b_group')
evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'count', GaussianNB(), 2000, 'cm_nb_c_group')
evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'tfidf', GaussianNB(), 2000, 'cm_nb_t_group')
evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'freq', GaussianNB(), 2000, 'cm_nb_f_group')

#evaluation_model(train_docs, test_docs, ytrain, ytest, 'binary', SVC(kernel = 'linear', random_state = 0), 'cm_svm_b_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'count', SVC(kernel = 'linear', random_state = 0), 'cm_svm_c_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'tfidf', SVC(kernel = 'linear', random_state = 0), 'cm_svm_t_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'freq', SVC(kernel = 'linear', random_state = 0), 'cm_svm_f_group')
#
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'binary', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2), 'cm_knn_b_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'count', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2), 'cm_knn_c_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'tfidf', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2), 'cm_knn_t_group')
#evaluation_model(train_docs, test_docs, ytrain, ytest, 'freq', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2), 'cm_knn_f_group')

# use diffrent label methode

#X, folders, lenght = get_all_folders()
## make labels
#y = make_labels_all(folders, lenght)
## Shuffle the whole set
#X, y = shuffle_sets(X,y)
##make training set and test set
#train_docs, test_docs, ytrain, ytest = train_test_split(X, y, 0.8)
#
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'binary', SGDClassifier(), 500, 'cm_lr_b_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'count', SGDClassifier(), 500, 'cm_lr_c_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'tfidf', SGDClassifier(), 500, 'cm_lr_t_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'freq', SGDClassifier(), 500, 'cm_lr_f_all')
#
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'binary', GaussianNB(), 500, 'cm_nb_b_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'count', GaussianNB(), 500, 'cm_nb_c_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'tfidf', GaussianNB(), 500, 'cm_nb_t_all')
#evaluation_model_partiel_fit(train_docs, test_docs, ytrain, ytest, 'freq', GaussianNB(), 500, 'cm_nb_f_all')