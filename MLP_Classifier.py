import numpy as np
from sklearn import neural_network, metrics
from sklearn.model_selection import train_test_split
import h5py
import pickle

f = h5py.File('lms_fun_v3.hdf5', 'r')
mismatched_v = np.asarray(f['mismatched_v'])
V = np.reshape(mismatched_v, [300600, 3])
mismatched_y = np.asarray(f['mismatched_y'])
y = np.reshape(mismatched_y, [300600])

#tried archs - 5x5, 6x2, 8x2, alpha = 0.001, 2x3(best testing error), add archs, solvers, lrs below for testing
archs = [(2,4)]
solvers = ['adam']
lrs = [0.0001]
best_arch = ''
best_lr = 0.0
best_solver = ''
for arch in archs:
    for solver in solvers:
        for lr in lrs:
            train_X, test_X, train_y, test_y = train_test_split(V, y, test_size = 0.5, random_state = 1, shuffle = True)
            model = neural_network.MLPRegressor(hidden_layer_sizes = arch, activation = 'relu', solver = solver, alpha = lr, random_state = 1)
            #train the MLP on train set
            model.fit(train_X, train_y)
            # model.set_params()
            #predict on test set
            predict_y = model.predict(test_X)
            error = metrics.mean_squared_error(test_y, predict_y)

            print("Architecture: " + str(arch) + " Solver: " + str(solver) + " Learning rate: %f Error: %f"%(lr, 100 * error))
            pickle.dump(model, open('nn.pkl','wb'))