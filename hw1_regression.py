
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from numpy.linalg import solve, inv
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# import the ridge regression form scikit learn

from sklearn.linear_model import Ridge

# This code implements the K-means clustering.
from numpy import genfromtxt


# read out command line arguments
lamb = float(sys.argv[1]) 
sigma2 = float(sys.argv[2])
X_train = genfromtxt(sys.argv[3], delimiter=',') 
y_train = genfromtxt(sys.argv[4], delimiter=',')
X_test = genfromtxt(sys.argv[5], delimiter=',')


#X_train = X_train[:, np.newaxis]


# do the sklearn ridge regression
n_samples, n_features = 10, 2
np.random.seed(0)
clf = Ridge(alpha=1/lamb)
clf.fit(X_train, y_train) 


#lamba sigma2 X_train.csv y_train.csv X_test.csv =  sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],delimiter=',')

#sigma2 = float(sigma2)
#lamb   = float(lamb)
'''
lamb = 1
X_train = np.array([258, 270, 294, 320, 342, 368, 396, 446, 480, 586])
X_train = np.vstack((X_train, np.ones(X_train.shape[0])))
X_train = np.transpose(X_train)
'''
# get the dimensions
[n,d] = X_train.shape # n samples, d dimensions 

'''
X_test = X_train
X_test[:,0] = X_train[:,0]*np.random.rand(n)*2
y_train = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368, 391.2, 390.8])
y_train = y_train[:, np.newaxis]
np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('X_test.csv',  X_test, delimiter = ',')
'''

# preprocess the data 

# subtract the mean
y_train_mean = np.mean(y_train)

#y_train = y_train-y_train_mean

# standardize the dimensions
X_means = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
#X_train = (X_train-X_means)/X_std

# calculate the linear ridge regression weights wRR
A = lamb*np.eye(d)+np.dot(np.transpose(X_train),X_train)
b = np.dot(np.transpose(X_train),y_train)
wRR = solve(A,b)
# print the wRR to the file
np.savetxt(''.join(['wRR_',str(int(lamb)),'.csv']), wRR, delimiter=',')





# part II: Active Learning

# calculate Sigma (the mu is the wRR)

# implement the active learning on page 61 in the script: 
maxIt = 10

n2 = X_test.shape[0]

sigma02 = np.zeros((n2,1))  
Xnew = X_test
Xmodel = X_train
ymodel = y_train
chosenIndices = np.zeros((1,10))



for i in range(maxIt):

	n2 = Xmodel.shape[0];
	# calculate the updated sigma matrix 
	Sigma = inv(lamb*np.eye(d)+1/sigma2*np.dot(np.transpose(Xmodel),Xmodel))

	# calculate the updated wRR (the mu)
	A = lamb * sigma2 * np.eye(d) + np.dot(np.transpose(Xmodel),Xmodel)
	b = np.dot(np.transpose(Xmodel),ymodel)
	wRR = solve(A,b)
	wRR = wRR[:,np.newaxis]
	# Incorporate a new datapoint
	# calculate the covariances for each new datapoint
	sigma02 = sigma2 + np.dot(np.dot(Xnew, Sigma),np.transpose(Xnew))
	sigma02 = np.diag(sigma02)


	# find the x0 with the highest sigma
	maxIndex = sigma02.argmax()
	Xmodel = np.vstack((Xmodel, Xnew[maxIndex,:]))
	ynew = np.dot(Xnew[maxIndex,:], wRR)
	ymodel = np.hstack((ymodel, ynew))

	originIndex = np.where((X_test == Xnew[maxIndex,:]).all(axis=1))
	chosenIndices[0,i] = originIndex[0][0]+1
	# delete the picked out datapoint from the array of new obervations
	Xnew = np.delete(Xnew, maxIndex, 0)

# write the indices to the file 
np.savetxt(''.join(['active_',str(int(lamb)),'_',str(int(sigma2)),'.csv']), chosenIndices, delimiter=',')

'''
plt.plot(X_train[:,0], y_train, marker='s', linestyle='None')
plt.plot(Xmodel[:,0], np.dot(Xmodel,wRR), marker ='<')
plt.show()
'''







#plt.plot(X_train, y_train+y_train_mean, marker='s')
#plt.plot(X_train, np.dot(X_train,wRR)+y_train_mean)
#plt.show()
