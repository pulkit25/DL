import numpy as np
import h5py

def relu(x: np.array) -> np.array:
    x = np.maximum(x, 0)
    return x

def softmax(x: np.array) -> np.array:
    s = np.exp(x) / np.sum(np.exp(x), axis = 0, keepdims = True)
    return s

def one_hot(y: np.array, num_classes: int) -> np.array:
    res = np.zeros([y.shape[0], num_classes])
    print(y)
    res[np.arange(y.shape[0]), y] = 1
    return res

def relu_derivative(z: np.array) -> np.array:
    res = np.zeros(z.shape)
    res[z >= 0] = 1
    return res

def init_params(parameters: dict, grad: dict, layer_size : list, L: int):
    for l in range(1, L):
        parameters['W%d'%l] = np.random.randn(layer_size[l], layer_size[l - 1]) * np.sqrt(2 / layer_size[l-1])
        parameters['b%d'%l] = np.zeros([layer_size[l], 1])
        grad['Vwt-1%d'%l] = 0
        grad['Vbt-1%d'%l] = 0

def feed_forward(parameters: dict, L: int, activation: str):
    if activation == 'tanh':
        for l in range(1, L - 1):
            parameters['z%d'%l] = np.dot(parameters['W%d'%(l)], parameters['a%d'%(l - 1)]) + parameters['b%d'%l]
            parameters['a%d'%l] = np.tanh(parameters['z%d'%l])
    elif activation == 'relu':
        for l in range(1, L - 1):
            parameters['z%d'%l] = np.dot(parameters['W%d'%(l)], parameters['a%d'%(l - 1)]) + parameters['b%d'%l]
            parameters['a%d'%l] = relu(parameters['z%d'%l])

    parameters['z%d'%(L - 1)] = np.dot(parameters['W%d'%(L - 1)], parameters['a%d'%(L - 2)]) + parameters['b%d'%(L - 1)]
    parameters['a%d'%(L - 1)] = softmax(parameters['z%d'%(L - 1)])

def back_propagation(grad: dict, parameters: dict, L: int, Y_batch: np.array, beta: float, mini_batch_size:int):
    grad['dZ%d'%(L - 1)] = parameters['a%d'%(L - 1)] - Y_batch.T
    grad['dW%d'%(L - 1)] = 1 / mini_batch_size * np.dot(grad['dZ%d'%(L - 1)], parameters['a%d'%(L - 2)].T)
    grad['db%d'%(L - 1)] = 1 / mini_batch_size * np.sum(grad['dZ%d'%(L - 1)], axis = 1, keepdims = True)
    grad['Vwt%d'%(L - 1)] = beta * grad['Vwt-1%d'%(L - 1)] + (1 - beta) * grad['dW%d'%(L - 1)]
    grad['Vbt%d'%(L - 1)] = beta * grad['Vbt-1%d'%(L - 1)] + (1 - beta) * grad['db%d'%(L - 1)]
    grad['Vwt-1%d'%(L - 1)] = grad['Vwt%d'%(L - 1)]
    grad['Vbt-1%d'%(L - 1)] = grad['Vbt%d'%(L - 1)]

    for l in range(L - 2, 0, -1):
        grad['dZ%d'%l] = np.dot(parameters['W%d'%(l + 1)].T, grad['dZ%d'%(l + 1)]) * relu_derivative(parameters['z%d'%l])
        grad['dW%d'%l] = 1 / mini_batch_size * np.dot(grad['dZ%d'%l], parameters['a%d'%(l - 1)].T)
        grad['db%d'%l] = 1 / mini_batch_size * np.sum(grad['dZ%d'%l], axis = 1, keepdims = True)
        grad['Vwt%d'%l] = beta * grad['Vwt-1%d'%l] + (1 - beta) * grad['dW%d'%l]
        grad['Vbt%d'%l] = beta * grad['Vbt-1%d'%l] + (1 - beta) * grad['db%d'%l]
        grad['Vwt-1%d'%l] = grad['Vwt%d'%l]
        grad['Vbt-1%d'%l] = grad['Vbt%d'%l]

def find_accuracy(X_val: np.array, Y_val: np.array, parameters: dict, L: int, activation: str) -> float:
    a = X_val.T
    if activation == 'tanh':
        for l in range(1, L - 1):
            z = np.dot(parameters['W%d'%(l)], a) + parameters['b%d'%l]
            a = np.tanh(z)
    elif activation == 'relu':
        for l in range(1, L - 1):
            z = np.dot(parameters['W%d'%(l)], a) + parameters['b%d'%l]
            a = relu(z)

    z = np.dot(parameters['W%d'%(L - 1)], a) + parameters['b%d'%(L - 1)]
    a = softmax(z)
    acc_vec = np.equal(np.argmax(a, axis = 0), np.argmax(Y_val.T, axis = 0))
    return np.sum(acc_vec) / acc_vec.shape[0]

class MLP:
    ctr = 0
    mini_batch_size = 50
    num_classes = 10
    X_train = []
    X_val = []
    Y_train = []
    Y_val = []
    X_test = []
    Y_test = []

    def __init__(self, file):
        self.ctr = 0
        data = np.asarray(file['xdata'])
        labels = np.asarray(file['ydata'])
        with h5py.File('mnist_testdata.hdf5') as f:
            self.X_test = np.asarray(f['xdata'])
            self.Y_test = np.asarray(f['ydata'])
        [self.X_train, self.X_val, self.Y_train, self.Y_val] = self.train_val_split(data, labels, val_size = 0, val_num_samples = 10000, shuffle = True)
        
    def train_val_split(self, data: np.array, labels: np.array, val_size: float = 0.1, val_num_samples: int = 0, shuffle: bool = True) -> tuple:
        #if splitting the data based on number of samples needed in val set, ignores the val_size param in the definition
        print("Splitting up data")
        n = data.shape[0]
        features = data.shape[1]

        if val_num_samples > 0:
            X_train = np.zeros([n - val_num_samples, features])
            Y_train = np.zeros([n - val_num_samples, 1])
            X_val = np.zeros([val_num_samples, features])
            Y_val = np.zeros([val_num_samples, 1])
            if shuffle:
                indices = np.random.permutation(np.arange(n))
                X_train = data[indices[val_num_samples:], :]
                X_val = data[indices[0: val_num_samples], :]
                Y_train = labels[indices[val_num_samples:], :]
                Y_val = labels[indices[0: val_num_samples], :]
            else:
                X_val = data[0: val_num_samples, :]
                X_train = data[val_num_samples: , :]
                Y_val = labels[0: val_num_samples, :]
                Y_train = labels[val_num_samples: , :]
            print('Training examples: %d and testing examples: %d'%(len(Y_train), len(Y_val)))
            return X_train, X_val, Y_train, Y_val
        #if splitting the data based on percentage of data
        else:
            X_train = np.zeros([n - val_size * n, features])
            Y_train = np.zeros([n - val_size * n, 1])
            X_val = np.zeros([val_size * n, features])
            Y_val = np.zeros([val_size * n, 1])
            if shuffle:
                indices = np.random.permutation(np.arange(n))
                X_train = data[indices[val_size * n:], :]
                X_val = data[indices[0: val_size * n], :]
                Y_train = labels[indices[val_size * n:] , :]
                Y_val = labels[indices[0: val_size * n], :]
            else:
                X_val = data[0: val_size * n, :]
                X_train = data[val_size * n: , :]
                Y_val = labels[0: val_size * n, :]
                Y_train = labels[val_size * n: , :]
            print('Training examples: %d and testing examples: %d'%(len(Y_train), len(Y_val)))
            return X_train, X_val, Y_train, Y_val
    
    def next_batch(self) -> np.array:
        self.ctr += 1
        return self.X_train[(self.ctr - 1) * self.mini_batch_size % self.X_train.shape[0]: (self.ctr) * self.mini_batch_size % self.X_train.shape[0], :], self.Y_train[(self.ctr - 1) * self.mini_batch_size % self.X_train.shape[0]: (self.ctr) * self.mini_batch_size % self.X_train.shape[0], :]

with h5py.File('mnist_traindata.hdf5') as file:
    mlp = MLP(file)

    #network params
    layer_size = [784, 512, 512, 10]
    L = len(layer_size)
    activation = 'tanh'
    learning_rate = 0.1
    initial_learning_rate = learning_rate
    num_epochs = 50
    beta = 0.98
    parameters = {}
    grad =  {}
    plot_x = []
    plot_y = {'train': [], 'test': []}
    cost = 10.0

    init_params(parameters, grad, layer_size, L)

    for epoch in range(num_epochs):
        #learning rate decay
        if epoch == 20 or epoch == 40:
            learning_rate /= 2
        for i in range(int(mlp.X_train.shape[0] / mlp.mini_batch_size)):
            #forward propagation
            X_batch, Y_batch = mlp.next_batch()

            parameters['a0'] = X_batch.T

            feed_forward(parameters, L, activation)

            cost = - 1 / mlp.mini_batch_size * np.sum(np.log(parameters['a%d'%(L - 1)]) * Y_batch.T)

            #back propagation
            back_propagation(grad, parameters, L, Y_batch, beta, mlp.mini_batch_size)

            #parameters update
            for l in range(1, L):
                parameters['W%d'%(l)] -= learning_rate * grad['Vwt%d'%l]
                parameters['b%d'%(l)] -= learning_rate * grad['Vbt%d'%l]

        print('Cost at epoch %d: %f' % (epoch, cost))
        train_acc = find_accuracy(mlp.X_train, mlp.Y_train, parameters, L, activation)
        test_acc = find_accuracy(mlp.X_test, mlp.Y_test, parameters, L, activation)
        print('Training accuracy: %f' % train_acc)
        print('Testing accuracy: %f' % test_acc)
        plot_x.append(epoch)
        plot_y['train'].append(train_acc)
        plot_y['test'].append(test_acc)

    with h5py.File('plot.hdf5','a') as f:
        if 'x' not in f.keys():
            f['x'] = plot_x
        if 'train_%s_%.3f'%(activation, initial_learning_rate) not in f.keys():
            f['train_%s_%.3f'%(activation, initial_learning_rate)] = plot_y['train']
        if 'test_%s_%.3f'%(activation, initial_learning_rate) not in f.keys():
            f['test_%s_%.3f'%(activation, initial_learning_rate)] = plot_y['test']

    with h5py.File('hw3p2sample.hdf5', 'w') as out:
        out.attrs['act'] = np.string_(activation)
        for i in range(1, L):
            out.create_dataset('w%d'%i, data = parameters['W%d'%i])
            out.create_dataset('b%d'%i, data = parameters['b%d'%i])