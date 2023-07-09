import numpy as np
from scipy import misc
from skimage.transform import resize
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering

re_size = 64  # ownsampling of resized rectangular image
img = misc.face(gray=True)  # retrieve a grayscale image
img = resize(img, (re_size, re_size))
mask = img.astype(bool)
graph = img_to_graph(img, mask=mask)
# Take a decreasing function of the gradient: we take it weakly
# dependant from the gradient the segmentation is close to a Voronoi.
graph.data = np.exp(-graph.data/graph.data.std())
labels = spectral_clustering(graph, n_clusters=3)
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.figure(figsize=(6,3))
plt.imshow(img, cmap='gray', interpolation='nearest')

plt.figure(figsize=(6,3))
plt.imshow(label_im, cmap=plt.cm.nipy_spectral, interpolation='nearest')

plt.show()
