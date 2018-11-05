#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:14:56 2018

@author: xaviervanegdom
"""
import csv

def open_text(name):
    with open(name, 'r', encoding = 'latin-1') as f:
        return f.read()

def save_all_chains(names, name, grams):
        for i in range(len(grams)):
            save_chain(f'grams/{names[i]}_{name}.csv', grams[i])

# create excelsheet
def save_chain(name, data):    
    with open(name, 'w', newline='') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(data)
 
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



#languages = ["NL", "EN"]
#triGrams = []
#biGrams = []
#for language in languages:
#    biGrams.append(calculate_all_n_gram(get_n_gram(open_text(f'{language}.txt'), 2)))
#    triGrams.append(calculate_all_n_gram(get_n_gram(open_text(f'{language}.txt'), 3)))
#            
#save_all_chains(languages, 'biGram', biGrams)
#save_all_chains(languages, 'triGram', triGrams)