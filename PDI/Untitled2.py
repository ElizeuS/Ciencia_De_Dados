
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans
from scikits.image.data import imread

im = imread('mio1.jpg')


X = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))


N_clus = 3
km = KMeans(k = N_clus, init = 'random')
km.fit(X.astype(float))

labels = np.reshape(km.labels_, im.shape[0:2])


pl.figure()
pl.imshow(im)
for l in range(N_clus):
    pl.contour(label == l, contours=1, \
               colors=[pl.cm.spectral(l / float(N_clus)), ])
pl.xticks(())
pl.yticks(())
pl.show()