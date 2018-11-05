#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:14:56 2018

@author: xaviervanegdom
"""
from math import log

def open_text(name):
    with open(name, 'r', encoding = 'latin-1') as f:
        return f.read()
# make a list of n grams with a string of words
def n_gram(words, n):
    nGrams = []
    for i in range(len(words) - (n - 1)):
        temp = []
        for j in range(n):
            temp.append(words[i + j])
        nGrams.append(temp)
    return nGrams        

# makes a ngram model of a dataset
def get_n_gram(word_data, n):   
    words = word_data.split()
    words = [word for word in words if word.isalpha()]
    return n_gram(words, n)
    
def calculate_all_n_gram(nGram):
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

def prediction(ngram, inputGram):
    p = 0
    for gram in inputGram:
        if gram in ngram[0]:
            p += log(ngram[1][ngram[0].index(gram)])
    return p

languages = ["NL", "EN"]
triGram = []
biGram = []
for language in languages:
    biGram.append(calculate_all_n_gram(get_n_gram(open_text(f'{language}.txt'), 2)))
    #triGram.append(calculate_all_n_gram(get_n_gram(open_text(f'{language}.txt'), 3)))

while True:
    inputText = input("input tekst: ") #"ik ben Xavier, ik ben moe"
    inputBiGram = get_n_gram(inputText, 2)
    #inputTriGram = get_n_gram(inputText, 3)
    for i in range(len(languages)):
        print(f'{languages[i]} Bi Gram: {prediction(biGram[i], inputBiGram)}')
        #print(f'{languages[i]} Tri Gram: {prediction(biGram[i], inputTriGram)}')