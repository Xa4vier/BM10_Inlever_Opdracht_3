#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:14:56 2018

@author: xaviervanegdom
"""
from save import save_grams
import Markov_chain as mc


languages = ["NL", "EN", "SE", "IT", "DE", "FR"]
triGrams = []
biGrams = []
for language in languages:
    biGrams.append(mc.get_markov_model(f'text/{language}.txt', 2))
    triGrams.append(mc.get_markov_model(f'text/{language}.txt', 3))
    print(f"{language} done!")

save_grams(languages, 'bigram', biGrams)
save_grams(languages, 'trigram', triGrams)