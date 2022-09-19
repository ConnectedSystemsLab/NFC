#!/usr/bin/env python3

"""Helper functions for NFC.
"""

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct
from scipy.interpolate import RectBivariateSpline
import scipy as sp
import os
import numpy as np
np.set_printoptions(precision=4)


def read_complex_binary2(filename):
    """ Read file of float32 into complex array.
    """

    with open(filename, 'rb') as f:
        bytes = f.read()
    data = np.frombuffer(bytes, dtype=np.float32).reshape(-1, 2)
    data = data[:, 0] + 1j*data[:, 1]
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
    # Grid is 0 to n
    f_mean = np.full((n+1, n+1), np.nan, dtype=np.float64)
    f_dev  = np.full((n+1, n+1), np.nan, dtype=np.float64)

    for x in tqdm(range(0,n+1,2)):
        for y in range(0,n+1,2):
            filename = os.path.join(dirname, f"{prefix}{x:>02}{y:>02}")
            rssis = get_rssis(filename)

            f_mean[x, y] = np.mean(rssis)
            f_dev[x, y] = 1.0

    return RectBivariateSpline(range(n+1), range(n+1), f_mean), RectBivariateSpline(range(n+1), range(n+1), f_dev), f_mean, f_dev


def log_normal_pdf(x, mean, stddev):
    """ Compute log of normal pdf.
    """
    return -0.5*np.log(2*np.pi*stddev**2) - (x-mean)**2/(2*stddev**2)


def calc_log_likelihood(rssis, x, y, f_mean, f_dev, c=0, n=0):
    """ Compute likelihood of observing 4 rssis measurements given the ground truth
        location is x, y for a model given by f_mean, f_dev.
    """

    log_likelihood = 0.0
    log_likelihood += log_normal_pdf(rssis[0], 
                                     f_mean(x, y) + c, f_dev(x, y))
    log_likelihood += log_normal_pdf(rssis[1],
                                     f_mean(n-y, x) + c, f_dev(n-y, x))
    log_likelihood += log_normal_pdf(rssis[2],
                                     f_mean(n-x, n-y) + c, f_dev(n-x, n-y))
    log_likelihood += log_normal_pdf(rssis[3],
                                     f_mean(y, n-x) + c, f_dev(y, n-x))
    # log_likelihood += sp.stats.norm.logpdf(rssis[0], 
    #                                        f_mean(x, y) + c, 
    #                                        f_dev(x, y))
    # log_likelihood += sp.stats.norm.logpdf(rssis[1],
    #                                        f_mean(n-y, x) + c, 
    #                                        f_dev(n-y, x))
    # log_likelihood += sp.stats.norm.logpdf(rssis[2],
    #                                        f_mean(n-x, n-y) + c, 
    #                                        f_dev(n-x, n-y))
    # log_likelihood += sp.stats.norm.logpdf(rssis[3],
    #                                        f_mean(y, n-x) + c, 
    #                                        f_dev(y, n-x))


    return log_likelihood


def localize_mle(rssis, f_mean, f_dev, c=0, n=0):
    """ Compute location estimate using maximum-likelihood estimation.
    """

    likelihood_grid = np.vectorize(calc_log_likelihood, excluded=['rssis', 'f_mean', 'f_dev', 'n'])(
        rssis=rssis,
        x=np.arange(0, n+0.1, 0.1),
        y=np.arange(0, n+0.1, 0.1)[:, np.newaxis],
        f_mean=f_mean,
        f_dev=f_dev,
        c=np.asarray(c)[:, np.newaxis, np.newaxis],
        n=n
    )

    return np.unravel_index(np.argmax(likelihood_grid), likelihood_grid.shape)

if __name__ == '__main__':
    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',   default='..',
                        help="Path to directory of data files.")
    parser.add_argument('--prefix',    type=int, default=0,
                        help="For each condition (air, water, oil, etc).")
    parser.add_argument('--grid_size', type=int,
                        default=10, help="Grid size as an integer.")
    args = parser.parse_args()
    print(args)

    n = args.grid_size  # grid is 0 to n (n+1 by n+1)

    # Compute the model from the directory.
    print('CREATING MODEL')
    f_mean, f_dev, f_mean_points, f_dev_points = get_model(args.dirname, args.prefix, n)

    # x, y = np.meshgrid(np.arange(0,n+0.1,0.1), np.arange(0,n+0.1,0.1))
    # z = f_mean(x, y, grid=False)
    x, y = np.meshgrid(np.arange(n+1), np.arange(n+1))
    z = f_mean_points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    plt.title('f_mean vs position')
    plt.show()


    # Create testing grid.
    print('CREATING TEST GRID')
    test_rssis = np.empty((n+1,n+1), dtype=np.float64)
    for x in tqdm(range(0,n+1,2)):
        for y in range(0,n+1,2):
            filename = os.path.join(args.dirname, f"1{x:>02}{y:>02}")
            rssis = get_rssis(filename)
            test_rssis[x, y] = np.mean(rssis)

    assert test_rssis.shape == f_mean_points.shape, f'shape no match {test_rssis.shape} {f_mean_points.shape}'

    # Start testing
    print('START TEST')
    errors = []
    for x in tqdm(range(0,n+1,2)):
        for y in range(0,n+1,2):
            rssis = [test_rssis[x, y], 
                     test_rssis[n-y, x],
                     test_rssis[n-x, n-y], 
                     test_rssis[y, n-x]]
            x0, y0, l = localize_mle(rssis, 
                                     f_mean, f_dev,
                                     range(-6, 0), 
                                     n)
            x0 /= 10.0
            y0 /= 10.0
            errors.append(np.sqrt((x-x0)**2+(y-y0)**2))
    print(np.mean(errors))
