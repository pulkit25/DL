import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

for alpha in [0.9, 0.5, 0.1, -0.5]:
    w, h = signal.freqz([1 - alpha], [1, -alpha])
    plt.plot(w / (2 * np.pi), np.abs(h), label = 'Alpha = ' + str(alpha))
plt.legend()

for alpha in [0.9, 0.5, 0.1]:
    print('Time  constant n: ' + str(np.log(0.2) / np.log(alpha)))

b, a = signal.butter(4, 0.25)
w, h = signal.freqz(b, a)
plt.plot(w / (2 * np.pi), np.abs(h), label = 'Magnitude of frequency response')

inp = np.random.randn(300)
out = signal.lfilter(b, a, inp)
plt.plot(range(300), inp, label = 'input')
plt.plot(range(300), out, label = 'output')
plt.legend()