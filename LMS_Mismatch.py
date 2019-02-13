import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('lms_fun_v3.hdf5', 'r')
# print(f.keys())
Vn = np.asarray(f['mismatched_v'])
Yn = np.asarray(f['mismatched_y'])

eta = 0.05
En = np.zeros((Vn.shape[0], Vn.shape[1]))
Wn = np.zeros(Vn.shape)
for i in range(Vn.shape[0]):
    wn = np.zeros((Vn.shape[2], 1))
    for j in range(Vn.shape[1]):
        vn = Vn[i,j,:].reshape((Vn.shape[2],1))
        yn = Yn[i,j].reshape((1,1))
        en = yn - np.dot(wn.T, vn)
        wn += eta * np.squeeze(en) * vn
        En[i, j] = np.squeeze(en)
        Wn[i, j, :] = wn.reshape((Vn.shape[2], ))

plt.plot(np.average(En, axis = 0))
plt.xlabel('Iterations')
plt.ylabel('Average MSE')
plt.figure()
Wavg = np.average(Wn, axis = 0)
plt.plot(Wavg)
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.show()