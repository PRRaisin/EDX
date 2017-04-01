Ist das angekommen?
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from numpy.linalg import solve
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# This code implements the K-means clustering.
from numpy import genfromtxt
#lamb, sigma2, X_train,  y_train, X_test  = genfromtxt(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],delimiter=',')

#sigma2 = float(sigma2)
#lamb   = float(lamb)

lamb = 1

X_train = np.array([258, 270, 294, 320, 342, 368, 396, 446, 480, 586])
X_train = X_train[:, np.newaxis]
y_train = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368, 391.2, 390.8])
y_train = y_train[:, np.newaxis]

print(X_train)



# get the dimensions
[n,d] = X_train.shape # n samples, d dimensions 

# preprocess the data 

# subtract the mean
y_train_mean = np.mean(y_train)
y_train = y_train-y_train_mean

# standardize the dimensions
X_means = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train-X_means)/X_std

# calculate the linear ridge regression weights wRR
A = lamb*np.eye(d)+np.dot(np.transpose(X_train),X_train)
b = np.dot(np.transpose(X_train),y_train)
wRR = solve(A,b)
print(wRR)
np.savetxt('wRR','.csv']), cent, delimiter=',')


plt.plot(X_train, y_train+y_train_mean, marker='s')
plt.plot(X_train, np.dot(X_train,wRR)+y_train_mean)
plt.show()


# print the wRR to the file

# apply wRR on the test data set. Remember to first subtract the old mean from the new data
#X_test_normalized = (X_test - X_means) / sigma2

#y_test = wRR * X_test_normalized






# write the results to the files
#np.savetxt(''.join(['centroids-',str(i+1),'.csv']), cent, delimiter=',')



#plt.scatter(X[:,0],X[:,1])
#plt.scatter(cent[:,0], cent[:,1], marker='s', color='g')
#plt.show()
