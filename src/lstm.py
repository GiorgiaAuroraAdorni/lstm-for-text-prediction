#!/usr/bin/env python3
#
# LSTM

import collections
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow.compat.v1 as tf


def check_dir(directory):
    """
    :param directory: path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def my_plot(train_dir, out_dir, model, epochs=5):
    """
    :param train_dir: OrderedDict containing the model and the correspondent validation accuracy
    :param out_dir: project directory for the output files
    :param model: name of the model
    :param epochs: number of epochs
    """
    check_dir(out_dir)
    train_loss = np.array([])

    with open(train_dir, 'r') as f:
        train_lines = f.readlines()

        for line in train_lines:
            el = line.strip('\n').split(',')
            train_loss = np.append(train_loss, float(el[1]))

    x = np.arange(1, epochs + 1, dtype=int)

    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)
    plt.xticks(x)

    plt.plot(x, train_loss, label='Train Loss')

    plt.legend()
    plt.title('Train Loss: model "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'loss.pdf')
    plt.show()
    plt.close()


def display_results():
    plot_dir = 'out/'
    model_dirs = ['initial/', 'dropout/', 'dropout-3layers/', 'preprocessed/', 'preprocessed-dropout/',
                  'preprocessed-dropout-3layers/', 'multibooks/', 'preprocessed-multibooks/', 'preprocessed-10epochs/',
                  'preprocessed-dropout-10epochs/', 'preprocessed-3layers/', 'preprocessed-3layers-10epochs/']
    train_losses = collections.OrderedDict()
    for model_dir in model_dirs:
        dir = plot_dir + model_dir
        set_dir = dir + 'train.txt'

        with open(set_dir, 'r') as f:
            train_lines = f.readlines()

        train_time = np.array([])
        train_loss = np.array([])

        for line in train_lines:
            el = line.strip('\n').split(',')
            train_time = np.append(train_time, float(el[2]))
            train_loss = np.append(train_loss, float(el[1]))

        total_train_time = np.sum(train_time)
        train_losses[set_dir] = train_loss[-1]

        print()
        print(model_dir)
        print('total_train_time: ', total_train_time)
        print('train_loss: ', train_loss[-1])
    print()
    train_losses = sorted(train_losses.items(), key=lambda el: el[1])
    print(train_losses)


def download_books_from_url(books_list, url_list):
    """
    :param books_list: list containing book titles
    :param url_list: list containing book urls
    """
    # Save the books to file
    for i, u in enumerate(url_list):
        r = requests.get(u, allow_redirects=True)

        check_dir('books/')
        with open('books/' + books_list[i] + '.txt', 'wb') as text_file:
            text_file.write(r.content.lower())

    # Define regex for the preprocessing of the files
    start_strings = ['volume one', '1 the three presents of d’artagnan the elder', 'chapter i.',
                     'chapter i.', '*the borgias*']
    # 'volume one', '1 the three presents', 'the borgia' occur multiple times in the book
    take_occurrence = [2, 2, 1, 1, 2]
    end_strings = ['footnotes', '――――', 'end of the man in the iron mask', 'end of ten years later', '————']
    removes = ['[0-9]+m', None, '\[[0-9]+\]', None, None]

    # Save preprocessed books
    for i in range(len(books_list)):
        with open('books/' + books_list[i] + '.txt', "r", encoding='utf-8') as reader:
            input_string = reader.read()

            input_string = preprocess_input(input_string, start_strings[i], end_strings[i], removes[i],
                                            take_occurrence[i])

            check_dir('books/preprocessed/')
            with open('books/preprocessed/' + books_list[i] + '.txt', 'w') as text_file:
                text_file.write(input_string)


def preprocess_input(input_string, start, end, remove, occurrence):
    """
    :param input_string: the input string
    :param start: a list containing the string by which each book starts
    :param end: a list containing the string by which each book ends
    :param remove: a list containing "stop words" to remove in each book
    :param occurrence: a list containing the number off occurrence of the starting regex
    :return input_string: updated input string
    """
    input_string = start + re.split(re.escape(start), input_string)[occurrence]
    input_string = re.split(end, input_string)[0]

    if remove is not None:
        input_string = re.sub(remove, '', input_string)

    input_string = re.sub('[\n]{3,}', '\n\n', input_string)

    return input_string


def create_dicts(input_string, model, statistics=False):
    """
    Convert characters to lower case, count the number of unique characters and the frequency of each character,
    choose one integer to represent each character and save the statistics in a LaTeX table.
    :param input_string: input text
    :param model: model name
    :param statistics: boolean that specify if save the statistics
    :return new_input_string, char_to_int, int_to_char, k, new_abs_freq, new_rel_freq: the updated input,
    two dictionaries to map characters to int and int to chars, the number of unique characters and the frequencies
    """
    input_string = np.array([c for c in input_string])
    abs_freq = collections.Counter(input_string)
    k1 = len(abs_freq)
    abs_freq = collections.OrderedDict(abs_freq)
    rel_freq = {key: value / len(input_string) for key, value in abs_freq.items()}

    char_to_int = {key: idx for idx, key in enumerate(abs_freq)}

    print('Initial dict size: ', k1)
    print('Initial input string length: ', len(input_string))
    if statistics:
        directory = 'out/' + model
        df = save_statistics(directory, 1, char_to_int, abs_freq, rel_freq)
        print('Initial frequencies: ', df)

    new_abs_freq = abs_freq.copy()

    # Substitute rare characters with unknown
    substituted_chars = []
    unk_val = 0
    for k, v in abs_freq.items():
        if v < 100:
            unk_val += v
            new_abs_freq['UNK'] = unk_val
            substituted_chars.append(k)
            new_abs_freq.pop(k)

    k = len(new_abs_freq)

    new_input_string = np.array([input_string], dtype=str)
    for sub_char in substituted_chars:
        idx = np.where(new_input_string[0] == sub_char)
        new_input_string[0, idx] = 'UNK'

    new_input_string = new_input_string[0].tolist()

    new_rel_freq = {key: value / len(new_input_string) for key, value in new_abs_freq.items()}

    char_to_int = {key: idx for idx, key in enumerate(new_abs_freq)}
    int_to_char = {idx: key for idx, key in enumerate(new_rel_freq)}

    print('Final dict size: ', k)
    if statistics:
        directory = 'out/' + model
        df = save_statistics(directory, 2, char_to_int, new_abs_freq, new_rel_freq)
        print('Final frequencies: ', df)

    print('Preprocessed input string length: ', len(new_input_string))
    return new_input_string, char_to_int, int_to_char, k, new_abs_freq, new_rel_freq


def save_statistics(directory, idx_table, char_to_int, abs_freq, rel_freq):
    """
    :param directory: directory where saves the statistics
    :param idx_table: index of the table
    :param char_to_int: dictionary that map characters to integers
    :param abs_freq: absolute frequencies
    :param rel_freq: relative frequencies
    :return df: data-frame containing all the statistics
    """
    idx = np.array(list(char_to_int.values()))
    chars = np.array(list(char_to_int.keys()))
    abs_freqs = np.array(list(abs_freq.values()))
    rel_freqs = np.array(list(rel_freq.values())) * 100
    rel_freqs_print = np.array([str(np.round(i, 3)) + '%' for i in rel_freqs])

    data = {'Encoding': idx,
            'Character': chars,
            'Absolute Frequencies': abs_freqs,
            'Relative Frequencies': rel_freqs_print}

    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Encoding')
    df = df.sort_values(by=['Absolute Frequencies'], ascending=False)

    check_dir(directory)
    with open(directory + '/MyTable' + str(idx_table) + '.tex', 'w') as latex_table:
        latex_table.write(df.to_latex())

    return df


def generate_batches(input_indices, batch_size, sequence_length):
    """
    The text is too long to allow backpropagation through time, so it must be broken down into smaller sequences.
    In order to allow backpropagation for a batch of sequences, the text may first be broken down into a number of
    large blocks, which corresponds to the batch size.

    Each of these blocks may be broken down further into subsequences, such that batch i contains the i-th subsequence
    of each block.
    During training, batches must be presented in order, and the state corresponding to each block must be preserved
    across batches.
    The technique described above is called truncated backpropagation through time.
    :param input_indices: input sequence (shape: (1))
    :param batch_size: number of blocks/batches in which divided the text
    :param sequence_length: the length of each subsequence of a block
    :return batches, mask: return the list of batches and the corresponding mask created after the padding
    """
    mod = 16 * 256
    input_dim = len(input_indices)

    missing = mod - (input_dim % mod)
    padded_input = np.pad(input_indices, [(0, missing)], mode='constant')

    blocks = np.reshape(padded_input, [batch_size, -1, sequence_length])
    batches = np.swapaxes(blocks, 0, 1)

    mask = np.concatenate([np.ones(input_dim), np.zeros(missing)])
    mask = np.reshape(mask, [batch_size, -1, sequence_length])
    mask = np.swapaxes(mask, 0, 1)

    return batches, mask


def create_network(hidden_units, num_layers, X, S, mask, dropout):
    """
    MultiRNNCell with two LSTMCells, each containing 256 units, and a softmax output layer with k units.
    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param num_layers: number of LSTM cells
    :param X: input placeholder
    :param S: placeholder for the LSTM state
    :param mask: placeholder for the mask (used during the computation of the loss)
    :param dropout: placeholder for the output_keep_prob of the dropout applied after all the LSTM cells
    :return Z, state: the output and the state of the network
    """
    k = tf.shape(X)[2]

    cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=n_units), output_keep_prob=dropout) for
            n_units in hidden_units]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cell)

    state_list = tf.unstack(S, axis=0)
    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_list[idx][0], state_list[idx][1])
                             for idx in range(num_layers)])

    sequence_length = tf.reduce_sum(mask, axis=1)
    rnn_outputs, state = tf.nn.dynamic_rnn(multi_cell,
                                           X,
                                           initial_state=rnn_tuple_state,
                                           sequence_length=sequence_length)

    # Softmax output layer with k units
    W = tf.Variable(tf.truncated_normal(shape=(hidden_units[0], k), stddev=0.1), name='W')
    b = tf.Variable(tf.zeros(shape=[k]), name='b')
    Z = tf.matmul(rnn_outputs, W) + b

    return Z, state


def net_param(hidden_units, learning_rate, num_layers, batch_size, seq_length, k):
    """

    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param learning_rate: learning rate
    :param num_layers: number of LSTM cells
    :param batch_size: batch size
    :param seq_length: length of the sequences
    :param k: vocab size
    :return X, Y, S, M, Z, state, loss, train: the state and the input placeholders and the the output of the network
    """
    with tf.variable_scope("model_{}".format(1)):
        X = tf.placeholder(tf.int32, [batch_size, seq_length], name='X')
        Y = tf.placeholder(tf.int32, [batch_size, seq_length], name='Y')
        S = tf.placeholder(tf.float32, [num_layers, 2, batch_size, hidden_units[0]], name='S')
        M = tf.placeholder(tf.float32, [batch_size, seq_length], name='M')
        dropout = tf.placeholder(tf.float32, [], name='dropout')

        # Transform the batch with one hot encoding
        X_onehot = tf.one_hot(X, depth=k)
        Y_onehot = tf.one_hot(Y, depth=k)

        Z, state = create_network(hidden_units, num_layers, X_onehot, S, M, dropout)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_onehot, logits=Z)
        loss = tf.reduce_sum(loss * M) / tf.reduce_sum(M)

        # Optimiser
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    return X, Y, S, M, Z, dropout, state, loss, train


def net_param_generation(hidden_units, num_layers, k):
    """

    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param num_layers: number of LSTM cells
    :param k: vocab size
    :return X, S, Z, dropout, state: the input and state placeholders: the output, the dropout placeholder and the
    state of the network
    """
    with tf.variable_scope("model_{}".format(1)):
        X = tf.placeholder(tf.int32, [20, 1], name='X')
        S = tf.placeholder(tf.float32, [num_layers, 2, 20, hidden_units[0]], name='S')
        M = tf.ones(X.shape[0:2])
        dropout = tf.placeholder(tf.float32, [], name='dropout')

        # Transform the batch with one hot encoding
        X_onehot = tf.one_hot(X, depth=k)

        Z, state = create_network(hidden_units, num_layers, X_onehot, S, M, dropout)

    return X, S, Z, dropout, state


def generate_sequences(int_to_char, char_to_int, num_sequence, seq_length, rel_freq, f_generation, hidden_units,
                       num_layers, config):
    """
    :param int_to_char: dictionary that maps integer to characters
    :param char_to_int: dictionary that maps characters to integer
    :param num_sequence: number of sequences to generate
    :param seq_length: length of the sequences
    :param rel_freq: relative frequencies
    :param f_generation: file writer
    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param num_layers: number of LSTM cells
    :param config: settings of the session
    :return sequences: the output sequences
    """
    tf.reset_default_graph()
    session2 = tf.Session(config=config)

    # Use the distribution of the output to generate a new character accordingly the distribution
    initial_chars = ''
    k = len(int_to_char)

    for i in range(num_sequence):
        char = np.random.choice(list(rel_freq.keys()), p=list(rel_freq.values()))
        initial_chars += char

    # Preprocess the input of the network
    encoded_input = [char_to_int[char] for char in initial_chars]
    encoded_input = np.expand_dims(encoded_input, axis=1)

    # Initialise and restore the network
    X, S, Z, dropout, state = net_param_generation(hidden_units, num_layers, k)

    Z_flat = tf.squeeze(Z)
    Z_indices = tf.random.categorical(Z_flat, num_samples=1)

    session2.run(tf.global_variables_initializer())

    new_saver = tf.train.Saver()
    new_saver.restore(session2, 'train/')

    # Generate sequences
    print("Starting generating…")
    gen_start = time.time()
    sequences = np.zeros(shape=[num_sequence, seq_length], dtype=str)

    current_state = np.zeros((num_layers, 2, num_sequence, hidden_units[0]))

    for j in range(seq_length):
        current_state, output = session2.run([state, Z_indices], feed_dict={X: encoded_input,
                                                                            S: current_state,
                                                                            dropout: 1.0})

        output = [int_to_char[s] for s in output.ravel()]

        sequences[:, j] = output

        encoded_input = [char_to_int[char] for char in output]
        encoded_input = np.expand_dims(encoded_input, axis=1)

    gen_end = time.time()
    gen_time = gen_end - gen_start

    f_generation.write('Generation Time, ' + str(gen_time) + '\n')
    print('Generation Time: {} sec.'.format(gen_time))

    for idx, seq in enumerate(sequences):
        seq = ''.join(seq)
        print('Sequence ', str(idx + 1), ': ', seq, '\n')
        f_generation.write('Sequence {}, {}\n'.format(idx + 1, seq))

    return sequences


def train_model(X_batches, Y_batches, batch_size, seq_length, k, epochs, hidden_units, learning_rate, d, mask,
                num_layers, config, model):
    """

    :param X_batches: input batches
    :param Y_batches: target batches
    :param batch_size: batch size
    :param seq_length: length of the sequences
    :param k: vocab size
    :param epochs: number of epochs for the training
    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param learning_rate: learning rate
    :param d: output_keep_prob of the dropout applied after all the LSTM cells
    :param mask: mask of the valid characters
    :param num_layers: number of LSTM cells
    :param config: settings of the session
    :param model: name of the model
    """
    # Create session
    tf.reset_default_graph()
    session = tf.Session(config=config)

    # Create model and set parameter
    X, Y, S, M, Z, dropout, state, loss, train = net_param(hidden_units, learning_rate, num_layers, batch_size,
                                                           seq_length, k)
    session.run(tf.global_variables_initializer())

    f_train = open('out/' + model + '/train.txt', "w")
    for e in range(0, epochs):
        print("Starting train…")
        train_start = time.time()
        print('Epoch: {}.'.format(e))

        cum_loss = 0
        cum_sum = 0
        current_state = np.zeros((num_layers, 2, batch_size, hidden_units[0]))

        for i in range(X_batches.shape[0]):
            batch_loss, _, current_state, output = session.run([loss, train, state, Z], feed_dict={X: X_batches[i],
                                                                                                   Y: Y_batches[i],
                                                                                                   S: current_state,
                                                                                                   dropout: d,
                                                                                                   M: mask[i]})

            cum_sum += np.sum(mask[i])
            cum_loss += batch_loss * np.sum(mask[i])
            print('Batch: ' + str(i) + '\tLoss: ' + str(batch_loss))

        epoch_loss = cum_loss / cum_sum

        train_end = time.time()
        train_time = train_end - train_start

        print('Train Loss: {:.2f}. Train Time: {} sec.'.format(epoch_loss, train_time))
        f_train.write(str(e) + ', ' + str(epoch_loss) + ',' + str(train_time) + '\n')

    f_train.close()
    saver = tf.train.Saver()
    saver.save(session, 'train/')


def main(download, preprocess, model, n_books, d=1.0, hidden_units=None, num_layers=2, epochs=5):
    """
    :param download: boolean that specifies if download the books
    :param preprocess: boolean that specifies if apply the preprocessing to the books
    :param model: model name
    :param n_books: number of books to use
    :param d: output_keep_prob of the dropout applied after all the LSTM cells [with default value 1]
    :param hidden_units: list containing the number of hidden units for each LSTM cell
    :param num_layers: number of LSTM cells
    :param epochs: number of epochs
    """
    # Download some books from Project Gutenberg in plain English text
    books_list = ['TheCountOfMonteCristo', 'TheThreeMusketeers', 'TheManInTheIronMask', 'TenYearsLater',
                  'CelebratedCrimes']
    url_list = ['http://www.gutenberg.org/files/1184/1184-0.txt', 'http://www.gutenberg.org/files/1257/1257-0.txt',
                'http://www.gutenberg.org/files/2759/2759-0.txt', 'http://www.gutenberg.org/files/2681/2681-0.txt',
                'http://www.gutenberg.org/files/2760/2760-0.txt']

    books_list = books_list[:n_books]
    url_list = url_list[:n_books]

    if download:
        download_books_from_url(books_list, url_list)

    inputs_string = ''

    if preprocess:
        path = 'books/preprocessed/'
    else:
        path = 'books/'

    for i in range(len(books_list)):
        with open(path + books_list[i] + '.txt', "r", encoding='utf-8') as reader:
            input_string = reader.read()
            inputs_string += input_string

    # Create dictionaries
    inputs_chars, char_to_int, int_to_char, k, abs_freq, rel_freq = create_dicts(inputs_string, model, statistics=True)

    # Encode the input
    encoded_input = [char_to_int[char] for char in inputs_chars]

    X = encoded_input[:-1]
    Y = encoded_input[1:]

    # Truncated backpropagation through time: use 16 blocks with subsequences of size 256.
    batch_size = 16
    sequence_length = 256

    # Create batches for samples and targets
    # _batches.shape: (batch_size, n_blocks, sequence_length)
    print('Generating batches…')
    X_batches, mask = generate_batches(X, batch_size, sequence_length)
    Y_batches, _ = generate_batches(Y, batch_size, sequence_length)
    print('Finished generating batches…')

    # Create configuration to avoid allocating all GPU memory upfront.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Training would take at least 5 epochs with a learning rate of 10^-2
    if hidden_units is None:
        hidden_units = [256, 256]

    learning_rate = 1e-2

    train_model(X_batches, Y_batches, batch_size, sequence_length, k, epochs, hidden_units, learning_rate, d, mask,
                num_layers, config, model)

    # Generate 20 sequences composed of 256 characters to evaluate the network
    num_sequence = 20
    seq_length = 256

    f_generation = open('out/' + model + '/generation.txt', "w")
    _ = generate_sequences(int_to_char, char_to_int, num_sequence, seq_length, rel_freq, f_generation, hidden_units,
                           num_layers, config)

    f_generation.close()


if __name__ == '__main__':
    # with open('/proc/self/oom_score_adj', 'w') as f:
    #     f.write('1000\n')

    # main(download=True, preprocess=False, model='initial', n_books=1)
    # main(download=True, preprocess=True, model='preprocessed', n_books=1)
    # main(download=True, preprocess=False, model='dropout', n_books=1, d=0.5)
    # main(download=True, preprocess=False, model='dropout-3layers', n_books=1, d=0.5,
    #      hidden_units=[256, 256, 256], num_layers=3)
    # main(download=True, preprocess=True, model='preprocessed-dropout', n_books=1, d=0.5)
    # main(download=True, preprocess=True, model='preprocessed-dropout-3layers', n_books=1, d=0.5,
    #      hidden_units=[256, 256, 256], num_layers=3)
    # main(download=True, preprocess=True, model='preprocessed-multibooks', n_books=3)
    # main(download=True, preprocess=False, model='multibooks', n_books=3)
    # main(download=True, preprocess=True, model='preprocessed-10epochs', n_books=1, epochs=10)
    # main(download=True, preprocess=True, model='preprocessed-dropout-10epochs', n_books=1, d=0.5, epochs=10)
    # main(download=True, preprocess=True, model='preprocessed-3layers', n_books=1, hidden_units=[256, 256, 256],
    #      num_layers=3)
    # main(download=True, preprocess=True, model='preprocessed-3layers-10epochs', n_books=1, hidden_units=[256, 256, 256],
    #      num_layers=3, epochs=10)

    # my_plot('out/initial/train.txt', 'out/initial/img/', model='initial')
    # my_plot('out/dropout/train.txt', 'out/dropout/img/', model='dropout')
    # my_plot('out/dropout-3layers/train.txt', 'out/dropout-3layers/img/', model='dropout-3layers')
    # my_plot('out/preprocessed/train.txt', 'out/preprocessed/img/', 'preprocessed')
    # my_plot('out/preprocessed-dropout/train.txt', 'out/preprocessed-dropout/img/', 'preprocessed-dropout')
    # my_plot('out/preprocessed-dropout-3layers/train.txt', 'out/preprocessed-dropout-3layers/img/',
    #         model='preprocessed-dropout-3layers')
    # my_plot('out/preprocessed-multibooks/train.txt', 'out/preprocessed-multibooks/img/', 'preprocessed-multibooks')
    # my_plot('out/multibooks/train.txt', 'out/multibooks/img/', model='multibooks')
    # my_plot('out/preprocessed-10epochs/train.txt', 'out/preprocessed-10epochs/img/', 'preprocessed-10epochs', epochs=10)
    # my_plot('out/preprocessed-dropout-10epochs/train.txt', 'out/preprocessed-dropout-10epochs/img/',
    #         'preprocessed-dropout-10epochs', epochs=10)
    # my_plot('out/preprocessed-3layers/train.txt', 'out/preprocessed-3layers/img/', 'preprocessed-3layers')
    # my_plot('out/preprocessed-3layers-10epochs/train.txt', 'out/preprocessed-3layers-10epochs/img/',
    #         'preprocessed-3layers-10epochs', epochs=10)

    display_results()
