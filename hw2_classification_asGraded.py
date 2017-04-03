
#from sklearn.datasets import make_blobs
from os.path import join
import numpy as np 
import sys

from numpy.random import choice, randint
from numpy.linalg import solve, inv, norm, det

from scipy.spatial.distance import cdist

# This code implements the K-means clustering.
from numpy import genfromtxt



# read out command line arguments
X_train = genfromtxt(sys.argv[1], delimiter=';') 
y_train = genfromtxt(sys.argv[2], delimiter=',')
X_test = genfromtxt(sys.argv[3], delimiter=',')


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
	Sigma[j,:,:] = 1/n_y[j]*np.dot(np.transpose(X_train[classIndices]-classMu), X_train[classIndices]-classMu)
	j = j+1

classPrior = n_y/n

# now, go for P(X=x|Y=y)*P(Y=y) on some new data X_test and y_test
n2 = X_test.shape[0]
f = np.zeros((n2,k))

for i in range(k):

	SigmaDiag = np.diag(np.diag(Sigma[i,:,:]))
	absSigma  = det(Sigma[i,:,:])
	SigmaInv = inv(Sigma[i,:,:])

	for m in range(0,n2):
		X = X_test[m,:]
		diff = X-mu[i,:]	
		expTerm = -1/2*np.dot(np.dot(np.transpose(diff), SigmaInv), diff)
		f[m,i] = classPrior[i]*absSigma**(-1/2)*np.exp(expTerm)
	

fsum = np.sum(f,1)
fsum = fsum[:, np.newaxis]
f = f/fsum

np.savetxt('probs_test.csv', f, delimiter=',')








