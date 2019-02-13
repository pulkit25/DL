import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('lms_fun_v3.hdf5', 'r')
#print(f.keys())
Vn = np.asarray(f['matched_3_v'])
Zn = np.asarray(f['matched_3_z'])
eta = 0.4
Wn = np.zeros(Vn.shape)
MSE = np.zeros((Vn.shape[0], Vn.shape[1]))

for i in range(Vn.shape[0]):
    wn = np.zeros((Vn.shape[2], 1))
    for j in range(Vn.shape[1]):
        vn = Vn[i,j,:].reshape((3,1))
        zn = Zn[i,j].reshape(1,1)
        en = zn - np.dot(wn.T, vn)
        MSE[i,j] = np.squeeze(en)
        wn += eta * np.squeeze(en) * vn
        Wn[i, j, :] = wn.reshape((3,))

Wavg = np.average(Wn, axis = 0)
plt.plot(Wavg[:,0])
plt.plot(Wavg[:,1])
plt.plot(Wavg[:,2])
plt.title('SNR = 3 dB, eta = %.2f'%eta)
plt.xlabel('Iterations')
plt.ylabel('Weights')

plt.figure()
squarederror = 10 * np.log10(np.average(np.square(MSE), axis = 0))
theoretical_mse = 10 * np.log10(0.5012)
plt.plot(squarederror)
plt.plot(theoretical_mse * np.ones(len(squarederror)), 'r--', label = 'LMMSE')
plt.title('SNR = 3 dB, eta = %.2f'%eta)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
# print(wn[:,0])
plt.show()