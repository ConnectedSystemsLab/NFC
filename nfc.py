#!/usr/bin/env python3

"""Helper functions for NFC.
"""

import os
import numpy as np
np.set_printoptions(precision=4)
import scipy as sp
import struct
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def read_complex_binary2(filename):
    """ Read file of float32 into complex array.
    """

    with open(filename, 'rb') as f:
        bytes = f.read()
    data = np.frombuffer(bytes, dtype=np.float32).reshape(-1,2)
    data = data[:,0] + 1j*data[:,1]
    return data

def get_rssis(filename):
    """ Get RSSI time series from file.
    """
    rssis = []
    data = read_complex_binary2(filename)
    num_frame = int(len(data)/65536 - 2)
    for i in range(num_frame):
        section = data[i*65536:(i+1)*65536]
        spectrum = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(section))**2))
        rssis.append(np.max(spectrum[22929:22949]))
    rssis = np.array(rssis)
    return rssis

def get_model(dirname, prefix, n):
    """ Get the likelihood model from directory of data files.
    """

    f_mean = np.zeros((n, n))
    f_dev  = np.zeros((n, n))
    
    for x in tqdm(range(n)):
        for y in range(n):
            filename = os.path.join(dirname, f"{prefix}{x:>02}{y:>02}")
            rssis = get_rssis(filename)

            f_mean[x,y] = np.mean(rssis)
            f_dev[x,y]  = np.std(rssis)

    return f_mean, f_dev


def calc_likelihood(rssis, x, y, f_mean, f_dev, c = 0):
    """ Compute likelihood of observing 4 rssis measurements given the ground truth
        location is x, y for a model given by f_mean, f_dev.
    """
    n = f_mean.shape[0]

    likelihood = 1.0
    likelihood *= sp.stats.norm.pdf(rssis[0], f_mean[x,y]         + c, f_dev[x,y])
    likelihood *= sp.stats.norm.pdf(rssis[1], f_mean[n-1-y,x]     + c, f_dev[n-1-y,x])
    likelihood *= sp.stats.norm.pdf(rssis[2], f_mean[n-1-x,n-1-y] + c, f_dev[n-1-x,n-1-y])
    likelihood *= sp.stats.norm.pdf(rssis[3], f_mean[y,n-1-x]       + c, f_dev[y,n-1-x])

    return likelihood

def localize_mle(rssis, f_mean, f_dev, c = 0):
    """ Compute location estimate using maximum-likelihood estimation.
    """
    n = f_mean.shape[0]

    likelihood_grid = np.ones((n,n))

    for x in range(n):
        for y in range(n):
            likelihood_grid[x,y] = calc_likelihood(rssis, x, y, f_mean, f_dev, c)

    return np.unravel_index(np.argmax(likelihood_grid), likelihood_grid.shape)


if __name__ == '__main__':

    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',   required=True, help="Path to directory of data files.")
    parser.add_argument('--prefix',    type=int, required=True, help="For each condition (air, water, oil, etc).")
    parser.add_argument('--grid_size', type=int, required=True, help="Grid size as an integer.")
    parser.add_argument('--query',     type=float, required=True, nargs=4, help="RSSI observation to localize.")
    args = parser.parse_args()

    # Compute the model from the directory.
    f_mean, f_dev = get_model(args.dirname, args.prefix, args.grid_size)

    # Visualize the model.
    x, y = np.meshgrid(range(f_mean.shape[0]), range(f_mean.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, f_mean)
    plt.title('f_mean vs position')
    plt.show()

    # Output MLE location for random RSSI observation.
    location = localize_mle(args.query, f_mean, f_dev)
    print(location)
    
