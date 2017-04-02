
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from pandas import read_excel
from numpy.random import choice, randint
from numpy.linalg import solve, inv, norm
from sklearn.cross_validation import train_test_split

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# This code implements the K-means clustering.
from numpy import genfromtxt



# read out command line arguments


X_train = genfromtxt(sys.argv[1], delimiter=';') 
y_train = genfromtxt(sys.argv[2], delimiter=',')
X_test = genfromtxt(sys.argv[3], delimiter=',')


# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.3, random_state=0)



n = y_train.shape[0]

k0 = int(min(y_train))
kmax = int(max(y_train))

k = kmax-k0+1
d = X_train.shape[1]

# calculate the class priors
n_y = np.zeros(k)
mu = np.zeros((k,d))
Sigma = np.zeros((k,d,d))

j=0
for i in range(k0, kmax+1):
	classIndices = np.where(y_train==i)
	n_y[j]=np.size(classIndices)
	classMu = np.mean(X_train[classIndices],0)
	mu[j,:] = classMu
	Sigma[j,:,:] = 1/n_y[-1]*np.dot(np.transpose(X_train[classIndices]-classMu), X_train[classIndices]-classMu)
	j = j+1

classPrior = n_y/n


# now, go for P(X=x|Y=y)*P(Y=y) on some new data X_test and y_test



f = np.zeros(k)
j=0
for i in range(k0, kmax+1):

	SigmaDiag = np.diag(Sigma[i,:,:])
	absSigma  = norm(SigmaDiag)
	#f[j] = classPrior[j]*absSigma**(-1/2)*np.exp(-1/2*np.dot(np.dot((X_test-mu[j,:]), inv(np.diag(SigmaDiag))), (X_test-mu[j,:])))
	A=np.dot(np.dot(X_test-mu[j,:], inv(Sigma[j,:,:])), np.transpose(X_test-mu[j,:]))
	print(A.shape)
	j = j+1


print(f)







