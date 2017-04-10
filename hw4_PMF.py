
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint, normal
from numpy.linalg import solve, norm
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from numpy import genfromtxt

# This code implements the PMF for assignment 4


# read the ratings data from the command line
ratings = genfromtxt(sys.argv[1], delimiter=',')

# to be sure, first transform them to real integers

ratingIndices = ratings[:,[0,1]].astype(int)

N1 = max(ratingIndices[:,0])
N2 = max(ratingIndices[:,1])

# build the matrix M
M = np.zeros((N1,N2))

for i in range(ratings.shape[0]):
	M[ratingIndices[i,0]-1, ratingIndices[i,1]-1] = ratings[i,2]


# set parameters to d=5, sigma^2=1/10 and lambda=2
d = 5
sigma2 = 0.1
lamb = 2.0 

maxIt = 50
eps = 10**-3


# generate the first v vector matrix from a normal distribution, v ~ N(0, lambda^-1), where dim(v) = d x N2
v = normal(0, 1/lamb, (N2,d))
u = np.zeros((N1,d))
L = np.zeros(maxIt)
### iterate ###
for it in range(maxIt):

	# update the u matrix (user location)
	for i in range(N1):

		A = lamb*sigma2*np.eye(d)
		b = np.zeros((1,d))
		# iterate over each rating of user i
		for j in range(N2):

			if norm(M[i,j]) > eps :

				A = A + np.outer(v[j,:], v[j,:])
				b = b + M[i,j]*v[j,:]

		u[i,:] = np.transpose(solve(A,np.transpose(b)))


	# update the v matrix (object location)
	# iterate over each object location vector
	for i in range(N2):

		A = lamb*sigma2*np.eye(d)
		b = np.zeros((1,d))

		# iterate over each user for a given object i
		for j in range(N1):

			if norm(M[j,i]) > eps :
				A = A + np.outer(u[j,:], u[j,:])
				b = b + M[j,i]*u[j,:]

		v[i,:] = np.transpose(solve(A,np.transpose(b)))

	# calculate the objective function
	
	# iterate over rating
	for i in range(ratings.shape[0]):
		L[it] = L[it] - 1./(2*sigma2) * norm(ratings[i,2]-np.dot(u[ratingIndices[i,0]-1,:], v[ratingIndices[i,1]-1,:]))**2 

	for i in range(N1):
		L[it] = L[it] - lamb/2*norm(u[i,:])**2

	for j in range(N2):
		L[it] = L[it] - lamb/2*norm(v[j,:])**2 


	save_its = np.array([10,25,50])-1

	if it in save_its:
		# write the results to the files
		np.savetxt(''.join(['U-',str(it+1),'.csv']), u, delimiter=',')
		np.savetxt(''.join(['V-',str(it+1),'.csv']), v, delimiter=',')


np.savetxt('objective.csv', L , delimiter=',')















