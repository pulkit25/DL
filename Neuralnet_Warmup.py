import numpy as np
import h5py
import matplotlib.pyplot as plt
import json

params_file = h5py.File('mnist_network_params.hdf5', 'r')
W1 = np.asarray(params_file['W1'])
W2 = np.asarray(params_file['W2'])
W3 = np.asarray(params_file['W3'])
b1 = np.reshape(np.asarray(params_file['b1']), [-1,1])
b2 = np.reshape(np.asarray(params_file['b2']), [-1,1])
b3 = np.reshape(np.asarray(params_file['b3']), [-1,1])

test_file = h5py.File('mnist_testdata.hdf5', 'r')
image_data = np.asarray(test_file['xdata'])
# labels_data = np.asarray(test_file['ydata'])

def relu(x):
    x = np.maximum(x, 0)
    return x

def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x), axis = 0)
    return s

def predict(x):
    a1 = relu(np.dot(W1, x.T) + b1)
    a2 = relu(np.dot(W2, a1) + b2)
    y = softmax(np.dot(W3, a2) + b3)
    return y

preds = predict(image_data)
outputs = []
for i in range(preds.shape[1]):
    outputs.append({'index': i, 'classification': int(np.argmax(preds[:, i])), 'activations': preds[:, i].tolist()})

print("AUTOGRADE: %s"%(json.dumps(outputs)))