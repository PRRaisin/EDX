
from os.path import join
import numpy as np 
import sys
import csv
from numpy.random import choice, randint
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import copy
#from matplotlib.patches import Ellipse
#from sklearn.datasets import make_blobs
#import matplotlib.pyplot as plt
#from matplotlib.mlab import bivariate_normal



def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



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

# chose the first k centroids randomly (also assign the same )
initialCent = X[randint(n, size=k), :]
cent = initialCent 


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


### Part II: The EM Algorithm ###
# Implementation is according to the description on page 194 of the script, on Gaussian Mixture models
prior = (1./k)*np.ones((k,1))
Sigma = np.zeros((k,d,d))

for i in range(k):
	Sigma[i,:,:] = np.eye(d)


mu = copy.deepcopy(initialCent)

#print(mu.shape)

for it in range(maxIt):

	# the E-step
	phi = np.zeros((n,k))

	# iterate over each sample x in X
	for j in range(n):

		multVarSum = 0
		# iterate over each cluster
		for m in range(k):
			pdf = multivariate_normal(mean = mu[m,:], cov = Sigma[m,:,:])
			phi[j,m] = prior[m]*pdf.pdf(X[j,:])
			
		#print(phi[j,:])
		#print(np.sum(phi[j,:]))
		#print('---------')
		phi[j,:] = phi[j,:]/np.sum(phi[j,:],0)

	# the M-step
	n_k = np.sum(phi,0)
	# update the prior
	prior = n_k/n	

	# initilize the mu and sigma vector
	mu = np.zeros((k,d))
	Sigma = np.zeros((k,d,d))
	
	for m in range(k):

		for i in range(n):
			mu[m,:] =  mu[m,:] + phi[i,m]*X[i,:] 
			
		mu[m,:] = mu[m,:]/n_k[m]


		#print(mu[m,:])
		for i in range(n):

			#print(diff.shape)
			Sigma[m,:,:] = Sigma[m,:,:] + phi[i,m] * np.outer(X[i,:] - mu[m,:], X[i,:] - mu[m,:] )


		#print(Sigma[m,:,:])
		Sigma[m,:,:] = Sigma[m,:,:]/n_k[m]


		# save the covariance matrix
		np.savetxt(''.join(['Sigma-',str(m+1),'-',str(it+1),'.csv']), Sigma[m,:,:], delimiter=',')

	#print(mu)
	# save the stuff
	np.savetxt(''.join(['pi-',str(it+1),'.csv']), prior, delimiter=',')
	np.savetxt(''.join(['mu-',str(it+1),'.csv']), mu, delimiter=',')

'''
## plot some results ##
plt.figure()
ax = plt.gca()
plt.scatter(X[:,0],X[:,1])
plt.scatter(cent[:,0], cent[:,1], marker='d', color='g')
plt.scatter(mu[:,0], mu[:,1], marker='s', color = 'y')
#for i in range(k):
#	plot_cov_ellipse(Sigma[i,:,:], mu[i,:], nstd=1, ax=ax)
plt.show()
'''


