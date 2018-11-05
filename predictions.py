# -*- coding: utf-8 -*-

from Markov_chain import get_ngram 
from load import load_chain 
from math import log
import time

def timer(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f' took: {te-ts:2.4f} sec')
        return result
    
    return timed

@timer
def prediction(ngram, inputGram):
    p = 0
    for gram in inputGram:
        if gram in ngram[0]:
            p += log(ngram[1][ngram[0].index(gram)])
    return p

languages = languages = ["NL", "EN", "SE", "IT", "DE", "FR"]
biGrams = []
triGrams = []
for language in languages:
    biGrams.append(load_chain(f'grams/bigram_{language}', 2))
    triGrams.append(load_chain(f'grams/trigram_{language}', 3))

loop = True
while loop:
    inputText = input("input tekst: ")
    if inputText == "q" : loop = False
    else :
        inputBiGram = get_ngram(inputText, 2)
        inputTriGram = get_ngram(inputText, 3)
        for i in range(len(languages)):
            print(f'{languages[i]} Bi Gram: {prediction(biGrams[i], inputBiGram)}')
            print(f'{languages[i]} Tri Gram: {prediction(triGrams[i], inputTriGram)}')