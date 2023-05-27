"""This module was adapted from the original implementation found here:
https://github.com/davidADSP/GDL_code/blob/master/models/RNNAttention.py

Which is a public repo under GNU General Public License v3.0
"""
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

from keras.layers import LSTM, Input, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import RepeatVector, Permute
from keras.layers import Multiply, Lambda
import keras.backend as K 
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import np_utils



def create_network(
    n_notes, n_durations, embed_size = 100, rnn_units = 256, use_attention = False
):
    """ create the structure of the neural network """

    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in) 

    x = Concatenate()([x1,x2])

    x = LSTM(rnn_units, return_sequences=True)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)

        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)
    
    else:
        c = LSTM(rnn_units)(x)

    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None

    opti = RMSprop(lr = 0.001)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model


def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)



def prepare_sequences(notes, durations, lookups, distincts, seq_len =32):
    """ Prepare the sequences used to train the Neural Network """

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)


def sample_with_temp(preds, temperature):

    if temperature == 0:
        return np.argmax(preds)
    else:
        np.random.seed(0)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)
