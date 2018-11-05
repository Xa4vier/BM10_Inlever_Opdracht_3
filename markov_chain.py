"""
Created on Mon Nov  5 12:59:29 2018

@author: xaviervanegdom
"""

# makes a ngram model of a dataset
def get_markov_model(name, n): 
    word_data = open_text(name)
    ngram = get_ngram(word_data, n)
    return calculate_all_ngram(ngram)

# split all words in a list and delete all not alpha
def get_ngram(word_data, n):
        words = word_data.split()
        words = [word for word in words if word.isalpha()]
        return ngram(words, n)
 
# make a list of n grams with a string of words
def ngram(words, n):
    nGrams = []
    for i in range(len(words) - (n - 1)):
        temp = []
        for j in range(n):
            temp.append(words[i + j])
        nGrams.append(temp)
    return nGrams      

# calculate the probabilities of the ngrams in the dataset
def calculate_all_ngram(nGram):
    temp = [] # how many times a ngram occurs 
    temp_words = [] # a list of all ngrams but now it will be a set 
    for gram in nGram:
        if gram in temp_words:
            temp[temp_words.index(gram)] += 1
        else :
            temp.append(1)
            temp_words.append(gram)
            
    return [temp_words, normalisation(temp)]

def normalisation(dataset):
    temp = []
    for i in dataset:
        temp.append(i / sum(dataset))
    return temp

def open_text(name):
    with open(name, 'r', encoding = 'latin-1') as f:
        return f.read()
