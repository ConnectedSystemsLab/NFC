#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from read_binary import read_complex_binary2

if __name__ == '__main__':
    rssi = []
    data = read_complex_binary2('00000')
    num_frame = int(len(data)/65536 - 2)
    for i in range(num_frame):
        section = data[i*65536:(i+1)*65536]
        spectrum = 10*np.log10(fftshift(np.abs(fft(section))**2))
        rssi.append(np.max(spectrum[22929:22949]))
    rssi = np.array(rssi)
    plt.plot(rssi)
    plt.show()
