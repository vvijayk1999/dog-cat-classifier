import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

IMG_SIZE = 50

pickle_in = open('x_pickle','rb')
x = pickle.load(pickle_in)

pickle_in = open('y_pickle','rb')
y = pickle.load(pickle_in)

x = x/255.0


dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = 'catsvdogs-convs-{}-nodes-{}-dense-{}-{}'.format(conv_layer,layer_sizes,dense_layer,int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for _ in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(x,y,batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

            model.save(NAME)