
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from numpy.linalg import solve, inv
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# This code implements the K-means clustering.
from numpy import genfromtxt


# read out command line arguments
lamb = float(sys.argv[1]) 
sigma2 = float(sys.argv[2])
X_train = genfromtxt(sys.argv[3], delimiter=',') 
y_train = genfromtxt(sys.argv[4], delimiter=',')
X_test = genfromtxt(sys.argv[5], delimiter=',')

