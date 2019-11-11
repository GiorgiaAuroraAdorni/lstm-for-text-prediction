#!/usr/bin/env python3
#
# LSTM

import numpy as np
import tensorflow as tf
import requests
import collections

# Download a large book from Project Gutenberg in plain English text
url = 'http://www.gutenberg.org/files/2701/2701-0.txt'
book = 'MobyDick.txt'
r = requests.get(url, allow_redirects=True)

# Preprocess the text
# convert characters to lower case
text = r.text.lower()
text_file = open(book, 'w')
n = text_file.write(text)
text_file.close()

with open(book, "r") as reader:
# count the number of unique characters and the frequency of each character
    input = reader.read()
    frequencies = collections.Counter(input)
    counter = len(frequencies)
    frequencies = collections.OrderedDict(frequencies)

    # choose one integer to represent each character
    dictionary = {tuple(key): idx for idx, key in enumerate(frequencies)}

