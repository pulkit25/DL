#Author: Pulkit Pattnaik
#Date: 16 February 2019
#Keras basic FC network
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras import optimizers
import matplotlib.pyplot as plt

with h5py.File('binary_random_sp2019.hdf5') as file:
    human_data = file['human'][:]
    machine_data = file['machine'][:]
    data = np.concatenate((human_data, machine_data))
    labels = np.zeros([data.shape[0], 1])
    labels[0:len(human_data)] = 1
    data, labels = shuffle(data, labels)

    model = Sequential(layers =
                       [Dense(20, input_shape = (20,)),
                        Activation('relu'),
                        Dense(1),
                        Activation('sigmoid')])

    model.compile(optimizer = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    history = model.fit(data, labels, validation_split = 0.2, epochs = 50, batch_size = 16)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.show()
    model.save('humanVmachine_adam.hdf5')