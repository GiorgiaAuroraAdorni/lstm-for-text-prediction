#!/usr/bin/env python3
#
# LSTM

import numpy as np
import tensorflow as tf
import requests
import collections


def generate_batches(input, batch_size, sequence_length, counter):
    """
    The text is too long to allow backpropagation through time, so it must be broken down into smaller sequences.
    In order to allow backpropagation for a batch of sequences, the text may first be broken down into a number of
    large blocks, which corresponds to the batch size.

    Each of these blocks may be broken down further into subsequences, such that batch i contains the i-th subsequence
    of each block.
    During training, batches must be presented in order, and the state corresponding to each block must be preserved
    across batches.
    The technique described above is called truncated backpropagation through time.
    :param text: input sequence
    :param batch_size: number of blocks/batches in which divided the text
    :param sequence_length: the length of each subsequence of a block
    :param counter:
    :return batches: return the list of batches
    """
    block_length = input.shape[0] // batch_size
    batches = np.array([]).reshape([0, batch_size, sequence_length, counter])

    for i in range(0, block_length, sequence_length):
        batch = np.array([]).reshape([0, sequence_length, counter])
        append = False
        for j in range(batch_size):
            start = j * block_length + i
            end = min(start + sequence_length, j * block_length + block_length)

            sequence = input[np.newaxis, start:end, :]

            if sequence.shape[1] == sequence_length:
                append = True
                batch = np.append(batch, sequence, axis=0)

        if append:
            batch = batch[np.newaxis, :, :, :]
            batches = np.append(batches, batch, axis=0)

    return batches


########################################################################################################################


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
    rel_freq = {key: value / len(input) for key, value in abs_freq.items()}

    # choose one integer to represent each character
    char_to_int = {key: idx for idx, key in enumerate(abs_freq)}
    int_to_char = {idx: key for idx, key in enumerate(abs_freq)}

    encoded_input = [char_to_int[char] for char in input]

    # One-hot encoding
    one_hot = tf.one_hot(encoded_input, depth=counter)
    X = one_hot[:-1]
    Y = one_hot[1:]

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    new_X, new_Y = session.run([X, Y])

    # # #
    # use 16 blocks with subsequences of size 256. In that case, the input tensor dimension is (16, 256, k), where k is
    # the number of unique characters (one-hot encoding).

    batch_size = 16
    sequence_length = 256

    X_batches = generate_batches(new_X, batch_size, sequence_length, counter)
    Y_batches = generate_batches(new_Y, batch_size, sequence_length, counter)

    print(X_batches, Y_batches)

print(counter)
print(abs_freq)
print(rel_freq)

    print(batches)
