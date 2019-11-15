#!/usr/bin/env python3
#
# LSTM

import numpy as np
import tensorflow as tf
import requests
import collections

# Download a large book from Project Gutenberg in plain English text

download = False
book = 'TheCountOfMonteCristo.txt'

if download:
    url = 'http://www.gutenberg.org/files/1184/1184-0.txt'
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
    input = np.array([c for c in input])

    abs_freq = collections.Counter(input)
    counter = len(abs_freq)
    abs_freq = collections.OrderedDict(abs_freq)

    rel_freq = {key: value / counter for key, value in abs_freq.items()}

    # choose one integer to represent each character
    char_to_int = {key: idx for idx, key in enumerate(abs_freq)}
    int_to_char = {idx: key for idx, key in enumerate(abs_freq)}

    integer_encoded = [char_to_int[char] for char in input]

    # One-hot encoding X_int
    X = tf.one_hot(integer_encoded, depth=counter)
    # One-hot encoding Y_int
    # Y = tf.one_hot(Y_int, depth=2) # shape: (batch_size, max_len, 2)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    new_X = session.run(X)

    # # #
    # use 16 blocks with subsequences of size 256. In that case, the input tensor dimension is (16, 256, k), where k is
    # the number of unique characters (one-hot encoding).

    batch_size = 16
    sequence_length = 256

    batches = generate_batches(new_X, batch_size, sequence_length)

    print(batches)
