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


def create_dicts(input, statistics=False):
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

    print(k)
    if statistics:
        dir = 'out/'
        df = save_statistics(dir, char_to_int, abs_freq, rel_freq)
        print(df)

    return input, char_to_int, int_to_char, k, abs_freq, rel_freq


def preprocessing(char_to_int, input):
    """

    :param char_to_int:
    :param input:
    :return:
    """
    encoded_input = [char_to_int[char] for char in input]

    return encoded_input


def one_hot_batches(encoded_input, k):
    one_hot = tf.one_hot(encoded_input, depth=k)
    X = one_hot[:-1]
    Y = one_hot[1:]

    return X, Y


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

    # softmax output layer with k units
    W = tf.Variable(tf.truncated_normal(shape=(hidden_units[0], k), stddev=0.1), name='W')
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


def net_param_generation():
    """

    :param num_sequence:
    :return S, X, Z, state:
    """
    with tf.variable_scope("model_{}".format(1)):
        X = tf.placeholder(tf.float32, [20, 1, 106], name='X')
        S = tf.placeholder(tf.float32, [num_layers, 2, 20, hidden_units[0]], name='S')

        Z, state = create_network(hidden_units, num_layers, X, S)

    return S, X, Z, state


def generate_sequences(int_to_char, char_to_int, num_sequence, seq_length, rel_freq, f_generation):
    tf.reset_default_graph()
    session2 = tf.Session(config=config)

    # num_sequence = 20
    # seq_length = 256
    k = len(int_to_char)

    # Use the distribution of the output to generate a new character accordingly the distribution
    initial_chars = ''

    for i in range(num_sequence):
        char = np.random.choice(list(rel_freq.keys()), p=list(rel_freq.values()))
        initial_chars += char

    # Preprocess the input of the network
    encoded_input = preprocessing(char_to_int, initial_chars)
    one_hot = tf.one_hot(encoded_input, depth=k)

    one_hot = session2.run(one_hot)

    input = np.expand_dims(one_hot, axis=1)

    # Initialise and restore the network
    S, X, Z, state = net_param_generation()

    Z_flat = tf.squeeze(Z)
    Z_indices = tf.random.categorical(Z_flat, num_samples=1)

    session2.run(tf.global_variables_initializer())

    new_saver = tf.train.Saver()
    new_saver.restore(session2, 'train/')

    # Generate sequences
    print("Starting generating…")
    gen_start = time.time()
    char_generated = np.zeros(shape=[num_sequence, seq_length], dtype=str)

    current_state = np.zeros((2, 2, num_sequence, hidden_units[0]))

    for j in range(seq_length):
        current_state, output = session2.run([state, Z_indices], feed_dict={X: input, S: current_state})

        output = [int_to_char[s] for s in output.ravel()]

        char_generated[:, j] = output

        encoded_input = preprocessing(char_to_int, output)
        one_hot = tf.one_hot(encoded_input, depth=k)
        one_hot = session2.run(one_hot)

        input = np.expand_dims(one_hot, axis=1)

    gen_end = time.time()
    gen_time = gen_end - gen_start

    f_generation.write('Generation Time, ' + str(gen_time))
    print('Generation Time: {} sec.'.format(gen_time))

    for idx, seq in enumerate(char_generated):
        print("Sequence: \n", seq)
        f_generation.write('Sequence ' + str(idx + 1) + ', ' + seq)

    return char_generated


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
    input, char_to_int, int_to_char, k, abs_freq, rel_freq = create_dicts(input, statistics=True)

    encoded_input = preprocessing(char_to_int, input)

    X, Y = one_hot_batches(encoded_input, k)

    # Create session and create configuration to avoid allocating all GPU memory upfront.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    writer = tf.summary.FileWriter("var/tensorboard/gio", session.graph)

    # Truncated backpropagation through time: use 16 blocks with subsequences of size 256.
    batch_size = 16
    sequence_length = 256

    # batches.shape: (646, 16, 256, 106)=(batch_size, n_blocks, sequence_length, vocab_size)
    X_batches = generate_batches(session.run(X), batch_size, sequence_length, k)
    Y_batches = generate_batches(session.run(Y), batch_size, sequence_length, k)
    print('Generated training batches')

    # Training would take at least 5 epochs with a learning rate of 10^-2
    hidden_units = [256, 256]
    num_layers = 2
    epochs = 5
    learning_rate = 1e-2

    # Create model and set parameter
    X, Y, S, Z, state, loss, train = net_param(hidden_units, learning_rate, num_layers)
    session.run(tf.global_variables_initializer())

    f_train = open('out/train.txt', "w")

    for e in range(0, epochs):
        print("Starting train…")
        train_start = time.time()
        print('Epoch: {}.'.format(e))

        cum_loss = 0
        current_state = np.zeros((2, 2, batch_size, hidden_units[0]))

        for i in range(X_batches.shape[0]):
            # Train
            train_loss, _, current_state, output = session.run([loss, train, state, Z],
                                                               feed_dict={X: X_batches[i],
                                                                          Y: Y_batches[i],
                                                                          S: current_state})

            cum_loss += train_loss
            print('batch: ' + str(i) + '\n\tloss: ' + str(train_loss))

        train_loss = cum_loss / X_batches.shape[0]

        train_end = time.time()
        train_time = train_end - train_start

        print('Train Loss: {:.2f}. Train Time: {} sec.'.format(train_loss, train_time))
        f_train.write(str(e) + ', ' + str(train_loss) + ',' + str(train_time) + '\n')

    f_train.close()

    saver = tf.train.Saver()
    saver.save(session, 'train/')

    # Generate 20 sequences composed of 256 characters to evaluate the network
    num_sequence = 20
    seq_length = 256

    f_generation = open('out/generation.txt', "w")
    sequences = generate_sequences(int_to_char, char_to_int, num_sequence, seq_length, rel_freq, f_generation)

    f_generation.close()
