import glob
import pickle
import numpy
import pandas as pd
import os
from music21 import converter, instrument, note, chord
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional, Flatten
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=100, type=int, help="Number of epoch")
parser.add_argument("--batch_size", default=16, type=int, help="Number of batch size")
parser.add_argument("--weight_dir", default=None, type=str, help="Weight directory")
args = parser.parse_args()
EPOCH_NUM = args.epoch
BATCH_SIZE = args.batch_size
WEIGHT_DIR = args.weight_dir
def train_network():
    """ Train a Neural Network to generate music """
    
    notes = pd.read_pickle(r"data/notes")
    
    if(notes is None):
        print("Preprocessing is required.")
        exit()
    print("Notes loaded")
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def prepare_sequences(notes, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        sequence_length = 50
    
        pitchnames = sorted(set(item for item in notes))
    
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
        network_input = []
        network_output = []
    
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
    
        n_patterns = len(network_input)
    
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(n_vocab)
    
        network_output = np_utils.to_categorical(network_output)
    
        return (network_input, network_output)

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(Bidirectional(LSTM(512,return_sequences=True),input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(512,return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(512,return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    if(WEIGHT_DIR is not None):
        model = load_model(WEIGHT_DIR,custom_objects={'SeqSelfAttention' : SeqSelfAttention})
    return model

def train(model, network_input, network_output):
    filepath = os.path.abspath("checkpoints/saved_model-{epoch:03d}-{loss:.4f}.tf")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
	    save_weights_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()