import pickle
import numpy as np

"""
Generate random data and save it as a .npy-file
"""

def generateData(size):
    """
    :param size: length of output array
    :return: numpy array with random numbers between 0 and 1, shape: (size, 2)
    """
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)
    return np.c_[x, y]

def saveData(data, filename='data.npy'):
    """
    Saves data in a file
    :param data: numpy array, data to be saved
    :param filename: filename of saved data
    """
    np.save(filename, data)

def loadData(filename):
    """
    Loads data from file
    """
    return np.load(filename)

if __name__ == '__main__':
    # simple example of how the code works
    data = generateData(100)
    saveData(data)

    data1 = loadData('data.npy')
    print(data1)