#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:14:56 2018

@author: xaviervanegdom
"""
from preprocessing import get_dataset_list
from math import log

def n_gram(words, n):
    biGrams = []
    for i in range(len(words) - (n - 1)):
        temp = []
        for j in range(n):
            temp.append(words[i + j])
        biGrams.append(temp)
    return biGrams        

def get_n_gram(word_data, n):    
    
    stringW = ""
    for i in word_data:
        stringW += ''.join(i) # make one big string off all the words
    
    words = stringW.split()
    return n_gram(words, n)
 
def calculate_all_n_gram(nGram):
    temp = []
    temp_words = []
    for gram in nGram:
        if gram in temp_words:
            temp[temp_words.index(gram)] += 1
        else :
            temp.append(1)
            temp_words.append(gram)
            
    return temp_words, temp

def normalisation(dataset):
    temp = []
    for i in dataset:
        temp.append(i / sum(dataset))
    return temp

wordsBi = get_n_gram(get_dataset_list('test.txt'), 2)
wordsBi, countBi = calculate_all_n_gram(wordsBi) 
countBi = normalisation(countBi)

BiGram = [wordsBi, countBi]

testText = "ik ben Xavier, ik ben moe"
testTextBiGram = get_n_gram(testText, 2)

p = 0
for gram in testTextBiGram:
    if gram in BiGram[0]:
        p += log(BiGram[1][BiGram[0].index(gram)])
      
    
          