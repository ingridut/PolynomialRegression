# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:59:58 2018
"""
from scipy import misc
import numpy as np
#from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



# Load the terrain
terrain1 = misc.imread('data.tif',flatten=0)
print(np.shape(terrain1))
# Show the terrain
plt.figure()
plt.title('Terrain over Ullevåll stadion - Sognsvann')
#plt.imshow(terrain1, cmap='gray')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()