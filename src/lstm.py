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


def create_network(hidden_units, num_layers, X, S):
    """
    MultiRNNCell with two LSTMCells, each containing 256 units, and a softmax output layer with k units.
    :param hidden_units:
    :param num_layer:
    :param X: input
    :param S: previous or initial state
    :return Z, state:
    """
    k = tf.shape(X)[2]

    cell = [tf.nn.rnn_cell.LSTMCell(num_units=n_units) for n_units in hidden_units]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cell)

    l = tf.unstack(S, axis=0)
    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)])

    rnn_outputs, state = tf.nn.dynamic_rnn(multi_cell, X, initial_state=rnn_tuple_state)
    # rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

    # softmax output layer with k units
    s = rnn_outputs.shape.as_list()[-1]

    W = tf.Variable(tf.truncated_normal(shape=(s, k), stddev=0.1), name='W')
    b = tf.Variable(tf.zeros(shape=[k]), name='b')
    Z = tf.matmul(rnn_outputs, W) + b

    return Z, state


def net_param(hidden_units, learning_rate, num_layers):
    """

    :param hidden_units:
    :param learning_rate:
    :param num_layers:
    :return:
    """
    with tf.variable_scope("model_{}".format(1)):
        X = tf.placeholder(tf.float32, [16, 256, 106], name='X')
        Y = tf.placeholder(tf.float32, [16, 256, 106], name='Y')
        S = tf.placeholder(tf.float32, [num_layers, 2, batch_size, hidden_units[0]], name='S')

        Z, state = create_network(hidden_units, num_layers, X, S)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
        loss = tf.reduce_mean(loss, name='loss')

        # Optimiser
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    return X, Y, S, Z, state, loss, train


########################################################################################################################


# Download a large book from Project Gutenberg in plain English text
download = False
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

    # Training would take at least 5 epochs with a learning rate of 10^-2
    hidden_units = [256, 256]
    num_layers = 2
    epochs = 5
    learning_rate = 1e-2

    # Create model and set parameter
    X, Y, S, Z, state, loss, train = net_param(hidden_units, learning_rate, num_layers)

    session.run(tf.global_variables_initializer())

    f_train = open('train.txt', "w")

    for e in range(0, epochs):
        print('Epoch: {}.'.format(e))

        avg_loss = 0

        print("Starting train…")
        train_start = time.time()

        for i in range(X_batches.shape[0]):
            # Train
            # FIXME: S, state
            init_state = np.zeros((2, 2, batch_size, hidden_units[0]))

            if i == 0:
                train_loss, _, current_state = session.run([loss, train, state], feed_dict={X: X_batches[i],
                                                                                            Y: Y_batches[i],
                                                                                            S: init_state})
                n_state = np.array(current_state)

            else:
                train_loss, _, current_state = session.run([loss, train, state], feed_dict={X: X_batches[i],
                                                                                            Y: Y_batches[i],
                                                                                            S: n_state})
                current_state = np.array(state)

            avg_loss += train_loss
            print('batch: ' + str(i) + '\n\tloss: ' + str(train_loss))

        train_loss = avg_loss / X_batches.shape[0]

        train_end = time.time()
        train_time = train_end - train_start

        print('Train Loss: {:.2f}. Train Time: {} sec.'.format(train_loss, train_time))
        f_train.write(str(epochs) + ', ' + str(train_loss) + ',' + str(train_time) + '\n')
