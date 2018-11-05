# -*- coding: utf-8 -*-

from markov_chain import get_n_gram 
from math import log
import time
import csv
import sys    

def timer(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f' took: {te-ts:2.4f} sec')
        return result
    
    return timed

def get_p(dataset): 
    p = dataset.split(',')
    p[0] = p[0].replace('[', '')
    p[len(p) - 1] = p[len(p) - 1].replace(']', '')
    return [float(i) for i in p]

def get_g2(dataset):
    g = dataset.split(',')
    for i in range(len(g)):
        g[i] = ''.join(ch for ch in g[i] if ch.isalnum())
    return [[g[i], g[i + 1]] for i in range(0, len(g) - 1, 2)]

def get_g3(dataset):
    g = dataset.split(',')
    for i in range(len(g)):
        g[i] = ''.join(ch for ch in g[i] if ch.isalnum())
    return [[g[i], g[i + 1], g[i + 2]] for i in range(0, len(g) - 1, 3)]

def load_chain(name, n):
    csv.field_size_limit(sys.maxsize)
    with open(f'grams/{name}.csv', 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    if n == 2:
        return [get_g2(dataset[0][0]), get_p(dataset[0][1])]
    else :
        return [get_g3(dataset[0][0]), get_p(dataset[0][1])]

@timer
def prediction(ngram, inputGram):
    p = 0
    for gram in inputGram:
        if gram in ngram[0]:
            p += log(ngram[1][ngram[0].index(gram)])
    return p

languages = ["NL", "EN", "ES", "IT", "DE", "FR"]
biGrams = []
triGrams = []
for language in languages:
    biGrams.append(load_chain(f'{language}_biGram', 2))
    triGrams.append(load_chain(f'{language}_triGram', 3))

loop = True
while loop:
    inputText = input("input tekst: ") #"ik ben Xavier, ik ben moe"
    if inputText == "q" : loop = False
    else :
        inputBiGram = get_n_gram(inputText, 2)
        inputTriGram = get_n_gram(inputText, 3)
        for i in range(len(languages)):
            print(f'{languages[i]} Bi Gram: {prediction(biGrams[i], inputBiGram)}')
            print(f'{languages[i]} Tri Gram: {prediction(triGrams[i], inputTriGram)}')