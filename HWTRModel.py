# Filename: HWTRModel.py
# Author: Dharmesh Tarapore <dharmesh@bu.edu>
# Description: A (very basic) handwriting transcription module.

import os
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Progbar

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape

class HWTRModel:

    def __init__(self,input_dims,universe_of_discourse):
        self.input_dims = input_dims
        self.universe_of_discourse = universe_of_discourse
        self.nn = hwtr
        # These are to use the ctc decode functionality outlined at:
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
        self.beam_width = 10
        self.top_paths = 1
        self.greedy = False
        self.model = None


    def summary(self):
        return self.model.summary()

    def compile(self, learning_rate=None):
    	'''
    	Override defined tf compile.
    	'''
    	outs = self.nn(self.input_dims, self.universe_of_discourse, learning_rate)
    	inputs, outputs, optimizer = outs
    	# create and compile
    	self.model = Model(inputs=inputs, outputs=outputs)
    	self.model.compile(optimizer=optimizer, loss=HWTRModel.custom_ctc_loss_function)
    
    def predict(self,x_val,batch_size=32,verbose=0,steps=1,callbacks=None,max_queue_size=10,
                workers=1,
                decode_using_ctc=True):
        '''
        Custom predict definition
        '''

        out = self.model.predict(x=x_val, batch_size=batch_size, verbose=verbose, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size)

        if not decode_using_ctc:
        	# Return negative log probabilities if decoding is not desired.
            return np.log(out)

        steps_done = 0

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        prediction, probabilities = [], []

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            # See: https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
            decode, log = tf.keras.backend.ctc_decode(np.asarray(out[index:until]),np.asarray([input_length for _ in range(len(x_test))]),
                                       greedy=self.greedy,beam_width=self.beam_width,top_paths=self.top_paths)

            probabilities.extend([np.exp(r) for r in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            prediction.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1

        return (prediction, probabilities)

    def fit(self, x,
            y,
            batch_size=32,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        self.model.fit(x,y, verbose = verbose)

    def evaluate(self,
        x,
        y,
        batch_size=32,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        ):
        return self.model.evaluate(x, y, verbose = verbose)
        
        
    def custom_ctc_loss_function(y_true, y_pred, *kwargs):
    	'''
    	CTC loss computation. Borrowed from the CTCModel.py file
    	with minor modifications.
    	'''
    	if len(y_true.shape) > 2:
    		y_true = tf.squeeze(y_true)
    	input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    	input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    	label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")
    	loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    	loss = tf.reduce_mean(loss)
    	return loss

def hwtr(input_dims, d_model, learning_rate):
    '''
    HWTR pipeline definition.
    Copied from the original definition in Draft.py
    '''
    input_data = Input(name="input", shape=input_dims)
    cnn = Reshape((input_dims[0] // 2, input_dims[1] // 2, input_dims[2] * 4))(input_data)
    cnn = Conv2D(filters=32, kernel_size=5, activation = 'relu', strides=1, padding="same")(input_data)
    cnn = Conv2D(filters=32, kernel_size=5, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = Conv2D(filters=32, kernel_size=5, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = Conv2D(filters=32, kernel_size=5, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = Conv2D(filters=64, kernel_size=3, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = Conv2D(filters=64, kernel_size=3, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = Conv2D(filters=8, kernel_size=3, activation = 'relu',strides=1, padding="same")(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=2, padding="valid")(cnn)

    reshape = Reshape((32,256))(cnn)

    rnn = Bidirectional(LSTM(256, input_shape = (32, 256), activation = 'relu', return_sequences=True))(reshape)
    rnn = Bidirectional(LSTM(256, activation = 'relu', return_sequences=True))(rnn)
    rnn = Bidirectional(LSTM(32, activation = 'relu', return_sequences=True))(rnn)
    rnn = Bidirectional(LSTM(80, activation = 'relu', return_sequences=True))(rnn)


    output_data = Dense(units=d_model, activation="softmax", name="lstm_output_matrix")(rnn)
    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)
