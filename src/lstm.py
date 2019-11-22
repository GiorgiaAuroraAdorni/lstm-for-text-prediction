#!/usr/bin/env python3
#
# LSTM

import numpy as np
import tensorflow.compat.v1 as tf
import requests
import collections
import time
import pandas as pd
import os


def check_dir(dir):
    """

    :param dir:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_statistics(dir, char_to_int, abs_freq, rel_freq):
    """

    :param dir:
    :param char_to_int:
    :param abs_freq:
    :param rel_freq:
    :return df:
    """
    idx = np.array(list(char_to_int.values()))
    chars = np.array(list(char_to_int.keys()))
    abs_freqs = np.array(list(abs_freq.values()))
    rel_freqs = np.array(list(rel_freq.values())) * 100
    rel_freqs_print = np.array([str(np.round(i, 3)) + '%' for i in rel_freqs])

    data = {'Encoding': idx,
            'Character': chars,
            'Absolute Frequence': abs_freqs,
            'Relative Frequence': rel_freqs_print}

    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Encoding')
    df = df.sort_values(by=['Absolute Frequence'], ascending=False)

    check_dir(dir)
    with open(dir + 'mytable.tex', 'w') as latex_table:
        latex_table.write(df.to_latex())

    return df


def preprocessing(input):
    """
    Convert characters to lower case, dount the number of unique characters and the frequency of each character,
    choose one integer to represent each character and save the statistics in a LaTeX table.
    :param input: input text
    :return char_to_int, int_to_char, encoded_input, k: two dictionaries to map characters to int and int to chars,
    the encoded input and the number of unique characters.
    """

    input = input.lower()

    input = np.array([c for c in input])
    abs_freq = collections.Counter(input)
    k = len(abs_freq)
    abs_freq = collections.OrderedDict(abs_freq)
    rel_freq = {key: value / len(input) for key, value in abs_freq.items()}

    char_to_int = {key: idx for idx, key in enumerate(abs_freq)}
    int_to_char = {idx: key for idx, key in enumerate(abs_freq)}

    encoded_input = [char_to_int[char] for char in input]

    print(k)
    dir = 'out/'
    df = save_statistics(dir, char_to_int, abs_freq, rel_freq)
    print(df)

    return char_to_int, int_to_char, encoded_input, k


def generate_batches(input, batch_size, sequence_length, k):
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
    :param k:
    :return batches: return the list of batches
    """
    mask = input.shape[0] - (input.shape[0] % (16 * 256))
    cropped_input = input[:mask, ...]

    blocks = np.reshape(cropped_input, [batch_size, -1, sequence_length, k])
    batches = np.swapaxes(blocks, 0, 1)

    return batches


def create_network(hidden_units, num_layer, X, init_state):
    """
    MultiRNNCell with two LSTMCells, each containing 256 units, and a softmax output layer with k units.
    :param hidden_units:
    :param num_layer:
    :param X:
    :param init_state:
    :return Z, Y_flat:
    """

    batch_size = X.shape[0]
    sequence_length = X.shape[1]
    k = X.shape[2]

    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layer)
    # multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

    if init_state == None:
        init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

    # FIXME final state never used
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, X, sequence_length=sequence_length,
                                                 initial_state=init_state)

    # rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

    # softmax output layer with k units
    # FIXME: hidden_units, 2   (2 is ot size)
    W = tf.Variable(tf.truncated_normal(shape=(hidden_units, k), stddev=0.1), name='W')
    b = tf.Variable(tf.zeros(shape=[k]), name='b')

    Z = tf.matmul(rnn_outputs, W) + b

    return Z, final_state


def net_param(hidden_units, num_layer, learning_rate, X, Y, init_state):
    Z, final_state = create_network(hidden_units, num_layer, X, init_state)

    # Loss function
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
    loss = tf.reduce_mean(loss, name='loss')

    # Optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    return Z, final_state, loss, train


########################################################################################################################


# Download a large book from Project Gutenberg in plain English text
download = True
book = 'TheCountOfMonteCristo.txt'

if download:
    url = 'http://www.gutenberg.org/files/1184/1184-0.txt'
    r = requests.get(url, allow_redirects=True)

    with open(book, 'wb') as text_file:
        text_file.write(r.content)

with open(book, "r", encoding='utf-8') as reader:
    input = reader.read()

    # preprocess data
    char_to_int, int_to_char, encoded_input, k = preprocessing(input)

    # One-hot encoding
    one_hot = tf.one_hot(encoded_input, depth=k)
    X = one_hot[:-1]
    Y = one_hot[1:]

    # Avoid allocating all GPU memory upfront.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    # Truncated backpropagation through time: use 16 blocks with subsequences of size 256.
    batch_size = 16
    sequence_length = 256

    # batches.shape: (646, 16, 256, 106)=(n_batches, n_blocks, seq_len, k)
    X_batches = generate_batches(session.run(X), batch_size, sequence_length, k)
    Y_batches = generate_batches(session.run(Y), batch_size, sequence_length, k)

    print(X_batches, Y_batches)


# You may use a MultiRNNCell with two LSTMCells, each containing 256 units, and a softmax output layer with k units.
hidden_units = sequence_length
num_layer = 2

Z, Y_flat = create_network(hidden_units, num_layer, batch_size, X)

# Training would take at least 5 epochs with a learning rate of 10^-2
epochs = 5
learning_rate = 1e-2

# Creates a mask to disregard padding
mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
mask = tf.reshape(mask, [-1])

# Network prediction
pred = tf.argmax(Z, axis=1) * tf.cast(mask, dtype=tf.int64)
pred = tf.reshape(pred, [-1, max_len]) # shape: (batch_size, max_len)
hits = tf.reduce_sum(tf.cast(tf.equal(pred, Y_int), tf.float32))
hits = hits - tf.reduce_sum(1 - mask) # Disregards padding

# Accuracy: correct predictions divided by total predictions
accuracy = hits/tf.reduce_sum(mask)

# Loss definition (masking to disregard padding)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
loss = tf.reduce_sum(loss*mask)/tf.reduce_sum(mask)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

