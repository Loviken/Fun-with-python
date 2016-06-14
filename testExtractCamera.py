import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plotBases(bases, title):
	nBases, sBase = bases.shape
	sideIm = np.sqrt(sBase)
	sideFi = np.sqrt(nBases)

	fig = plt.figure(title)
	for i in range(0, nBases):
		nSubPlt = sideFi * 110 + i + 1
		
		tmpFig = fig.add_subplot(nSubPlt)
		tmpFig.matshow(bases[i].reshape(sideIm, sideIm), cmap = 'gray', \
		vmin = 0.1, vmax = 0.9)

samples  = 500
sampSize = 10
nBases   = 9 # for PCA
nClust   = 100
#np.random.seed(1000) # Every new image is unique anyway

capture = cv2.VideoCapture(0)

#for i in range(1,100):
#	r,f = capture.read()
#	cv2.imshow("Camera", f)
#	cv2.waitKey(1) # mSec

# make gray-scale
r,f = capture.read()
capture.release()

theImage = f.mean(2)/255

hi, wi = theImage.shape

plt.matshow(theImage, cmap = 'gray')
#plt.set()


# Collect all the samples
points = np.empty((samples, sampSize**2))
for i in range(0,samples):
	coord = np.random.rand(2)
	x = int(coord[0]*(hi-sampSize))
	y = int(coord[1]*(wi-sampSize))
	
	tmpImage  = theImage[x:x + sampSize, x:x + sampSize]
	points[i] = tmpImage.ravel() # make 1D
	
#plt.figure(0)
#plt.plot(points[0],points[1],'.')

# PCA
pca = PCA(n_components=nBases)
pca.fit(points)
pca_points = pca.transform(points)

plt.figure()
plt.plot(pca_points[:,0],pca_points[:,1],'.')

# Create clusters
kMean = KMeans(nClust)
kMean.fit(points)
kCenters = kMean.cluster_centers_
kC_PCA   = pca.transform(kCenters)
plt.plot(kC_PCA[:,0], kC_PCA[:,1], 'r^')

# What does the eigenvectors look like?
invEye = pca.inverse_transform(np.eye(nBases))
# pca.components_ doesn't work. Why?
plotBases(invEye, 'Eigenvectors')

# What do the cluster centers look like?
#invK = pca.inverse_transform(kCenters)
#plotBases(kCenters, 'Clusters')

# Recreation
image2 = theImage.copy() # PCA space
image3 = theImage.copy() # Cluster space
for i in range(0,hi,sampSize):
	for j in range(0,wi,sampSize):
		patch = image2[i:i + sampSize, j:j + sampSize]
		trans_patch = pca.transform(patch.ravel())
		tmp = pca.inverse_transform(trans_patch)
		image2[i:i + sampSize, j:j + sampSize] = tmp.reshape(sampSize, sampSize)
		
		tmp2 = kMean.predict(patch.ravel())[0]
		image3[i:i + sampSize, j:j + sampSize] = kCenters[tmp2].reshape(sampSize, sampSize)

plt.matshow(image2, cmap = 'gray', vmin = 0, vmax = 1)
plt.matshow(image3, cmap = 'gray')

# Show all
plt.show()
