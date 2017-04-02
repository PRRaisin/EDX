
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# This code implements the K-means clustering.
from numpy import genfromtxt
X = genfromtxt(sys.argv[1], delimiter=',')



k = 5 # this is a given

maxIt = 10


# generate the data
#X,y = make_blobs(n_samples=150, n_features=2, centers=k, cluster_std=0.5, shuffle=True, random_state=0)
[n,d] = X.shape

#np.savetxt('X.csv',X, delimiter=',' )


# chose the first k centroids randomly 
cent = X[randint(n, size=k), :]

for i in range(maxIt):

	# calculate the distances to each cluster and find the min indices
	dist = cdist(X,cent,'euclidean')
	
	# find the lowest values and set them to the respective clusters
	clustInd = np.argmin(dist, axis=1)
	
	# calculate the new cluster centroids
	for j in range(k):
		cent[j,:] = np.mean(X[clustInd == j,:], axis=0)

	# write the results to the files
	np.savetxt(''.join(['centroids-',str(i+1),'.csv']), cent, delimiter=',')



plt.scatter(X[:,0],X[:,1])
plt.scatter(cent[:,0], cent[:,1], marker='s', color='g')
plt.show()
