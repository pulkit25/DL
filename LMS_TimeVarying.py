import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('lms_fun_v3.hdf5', 'r')
# print(f.keys())
Vn = np.asarray(f['timevarying_v'])
Zn = np.asarray(f['timevarying_y'])
Wtrue = np.asarray(f['timevarying_coefficents'])
eta = 0.15
Wn = np.zeros(Vn.shape)

wn = np.zeros((Vn.shape[1], 1))
for i in range(Vn.shape[0]):
    vn = Vn[i,:].reshape((Vn.shape[1],1))
    zn = Zn[i].reshape((1,1))
    en = zn - np.dot(wn.T, vn)
    wn += eta * np.squeeze(en) * vn
    Wn[i, :] = wn.reshape((Vn.shape[1],))

plt.plot(Wtrue)
plt.plot(Wn)
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.show()