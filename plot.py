#Author: Pulkit Pattnaik
#Date: 16 February 2019
#read hdf5 file and plot data
import matplotlib.pyplot as plt
import h5py
import numpy as np

with h5py.File('plot.hdf5', 'r') as f:
    plt.figure()
    train = 100 * np.asarray(f['train_relu_0.001'])
    test = 100 * np.asarray(f['test_relu_0.001'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.figure()
    train = 100 * np.asarray(f['train_relu_0.010'])
    test = 100 * np.asarray(f['test_relu_0.010'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.figure()
    train = 100 * np.asarray(f['train_relu_0.100'])
    test = 100 * np.asarray(f['test_relu_0.100'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.figure()
    train = 100 * np.asarray(f['train_tanh_0.100'])
    test = 100 * np.asarray(f['test_tanh_0.100'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.figure()
    train = 100 * np.asarray(f['train_tanh_0.100'])
    test = 100 * np.asarray(f['test_tanh_0.100'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.figure()
    train = 100 * np.asarray(f['train_tanh_0.100'])
    test = 100 * np.asarray(f['test_tanh_0.100'])
    min_train = int(np.min(train))
    plt.plot(f['x'], train)
    plt.plot(f['x'], test)
    rang = np.arange(min_train, 100)
    plt.plot(20 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.plot(40 * np.ones([rang.shape[0], 1]), rang, ':')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()