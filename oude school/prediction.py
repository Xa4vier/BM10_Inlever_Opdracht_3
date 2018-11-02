'''
 -*- coding: utf-8 -*-

Xavier van Egdom
Hogeschool Zuyd HBO-ICT

Predicting with Models
'''

from string import punctuation
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from keras.preprocessing.text import Tokenizer
import _pickle as cPickle
import pickle

# load the model
def load_model(name):
    with open(name + '.pkl', 'rb') as fid:
        #print('loading model successfully!')
        return cPickle.load(fid)
    
# tokenizer loading
def load_tokenizer(name):
    with open(name + '_tokenizer.pickle', 'rb') as handle:
        #print('loading tokenizer successfully!')
        return pickle.load(handle)

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

# classify a review
def predict_category(article, vocab, tokenizer, model, mode):
    # clean
    tokens = clean_doc(article)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode=mode)
    # prediction
    yi = model.predict(encoded)
    yf = model.predict_proba(encoded)
    yf = [float("{0:.2f}".format(p * 100)) for p in yf[0]]
    return yi, yf

def predict(article):
    # set mode
    mode = 'binary'
    name = "model_tfidf_LR_sport_loss_log"
    vocab = load_doc('vocab_trans.txt')
    vocab = vocab.split()
    vocab = set(vocab)    
    # loading model
    model = load_model("models/" + name)
    # loading tokenizer
    tokenizer = load_tokenizer("models/" + name)
    return predict_category(article, vocab, tokenizer, model, mode)

def print_prediction(yi, yf, labels):
    print('\ncategorie: ' + labels[yi[0]], end="\n\n")
    for text in labels:
        print(text + ", ", end="")
    print("\n" + str(yf))

while True:
    labels = ['Nederlands', 'Spanje']
    #get the articale
    article = input("tekst: ")
    #stop loop
    if article == "q": break
    #make prediction
    yi, yf = predict(article)
    # print prediction
    print_prediction(yi, yf, labels)
    #get the index of the probability list
    index = yf.index(max(yf))
    
