#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=4)
import struct

def read_complex_binary2(filename):
    with open(filename, 'rb') as f:
        bytes = f.read()
    data = np.frombuffer(bytes, dtype=np.float32).reshape(-1,2)
    data = data[:,0] + 1j*data[:,1]
    return data

if __name__ == '__main__':
    data = read_complex_binary2('00000')
    print(data[:10])
