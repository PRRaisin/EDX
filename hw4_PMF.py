
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# This code implements the K-means clustering.
from numpy import genfromtxt
X = genfromtxt(sys.argv[1], delimiter=',')

k = 5 # this is a given
maxIt = 10
# generate the data
#X,y = make_blobs(n_samples=150, n_features=2, centers=k, cluster_std=0.5, shuffle=True, random_state=0)
[n,d] = X.shape
#np.savetxt('X.csv',X, delimiter=',' )


### Part I: k-means clustering ###

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


'''
plt.scatter(X[:,0],X[:,1])
plt.scatter(cent[:,0], cent[:,1], marker='s', color='g')
plt.show()
'''




### Part II: The EM Algorithm ###
# Implementation is according to the description on page 194 of the script, on Gaussian Mixture models


prior = (1./k)*np.ones((k,1))
Sigma = np.zeros((k,d,d))
for i in range(k):
	Sigma[i,:,:] = np.eye(d)

mu = X[randint(n, size=k), :]

#print(mu.shape)

for i in range(maxIt):


	# the E-step
	phi = np.zeros((n,k))

	# iterate over each sample x in X
	for j in range(n):

		multVarSum = 0
		# iterate over each cluster
		for m in range(k):
			pdf = multivariate_normal(mean = mu[m,:], cov = Sigma[m,:,:])
			multVar = pdf.pdf(X[j,:])
			phi[j,m] = prior[m]*multVar
			multVarSum =multVarSum + multVar

		phi[j,:] = phi[j,:]/multVarSum

	# the M-step
	print(phi)
	n_k = np.sum(phi,0)
	print(n_k)
	# update the prior
	prior = n_k/n	
	
	for m in range(k):

		for i in range(n):
			mu[m,:] =  mu[m,:] + phi[i,m]*X[i,:] 

		#print(n_k[m])
		mu[m,:] = mu[m,:]/n_k[m]

		for i in range(n):
			Sigma[m,:,:] = Sigma[m,:,:] + phi[i,m] * np.dot(np.transpose(X[i,:] - mu[m,:]), (X[i,:] - mu[m,:]))















