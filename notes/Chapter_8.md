# Graph Neural Networks

1. [Basics of Graphs](#basics-of-graphs)
2. [Mathematics for Graphs](#mathematics-for-graphs)

## Basics of Graphs

- ### Overview

  - Graphs are a super general representation of data with intrinsic structure.
  - Images have a very strong notion of **locality**.
  - Image = Grid structure + channel intensities
    - It makes sense to design filters to group representation from a neighborhood of pixels. These filters are convolutions.
    - Each pixel has a vector of features that describe it.
      - The channel intensities can be regarded as the signal of the image.

- ### Decomposing features (signal) and structure

  - As seen in the chapter on [Transformers](./Chapter_7.md), natural language can also be decomposed to signal and structure.
    - Structure:
      - Order of words, which implies syntax and grammar context.
    - The features will be a set of word embeddings, and the order will be encoded in the positional embeddings.

- ### Real-world signals that we can model with graphs

## Mathematics for Graphs

- ### The basic maths for processing graph-strucuted data

  - Graph signal $X \in R^{N * F}$
  - Adjacency matrix $A \in R^{N * N}$
  - Degree of node:
    - The number of nodes connected to.
    - e.g. non-corner pixel has a degree of 8 (which is the surrounding pixels)
  - Degree matrix $D$:
    - Fundamental in graph theory
      - Provides a single value for each node.
    - Also used for the computation of the most important graph operator: the grpah Laplacian.

- ### The graph Laplacian
  
  - Defined as:
    - $L = D - A$
  - Normalized version is used in graph neural networks.
    - To avoid problems when processing with greapdient-based methods.
  - $L_{norm} = D^{-1/2} L D^{-1/2}$
    - $L_{norm} = I - D^{-1/2} A D^{-1/2}$
  - A slightly alternated version often used in graph neural networks:
    - $L_{norm} = D^{-1/2} (A + I) D^{-1/2}$
    - From now on, this will be referred as normalized graph laplacian.
  - With this trick, the input can be fed into a gradient-based algorithm without causing instabilities.
  - Implementation:
    - [Graph Laplacian](../code/graph_laplacian.py)

- ### Laplacian eigenvalues and eigenvectors

  - The multiplicity of the zero eigenvalue of the graph laplacian is equal to the number of connected components.
  - Additional info (KA):
    - [Petersen graph](https://en.wikipedia.org/wiki/Petersen_graph)
      - A small graph that serves as a useful example and counterexample for many problems in graph theory.
  - Implementation:
    - [Examples of connected graphs](../code/connected_components.py)

- ### Spectral image segmentation with graph Laplacian

  - In graph, the smallest non-zero eigenvalue has been used for "spectral" image segmentation from the 90s.
  - By converting a grayscale image to a graph, we can divide an image based on its slowest non-zero frequencies/eigenvalues.
  - Spectral segmentation is an unsupervised algorithm to segment an image based on the eignevalues of the laplacian.
  - Implementation:
    - [Spectral image segmentation](../code/spectral_image_segmentation.py)
      - Three clusters of raccoon's image.
  - Additional info (KA):
    - [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)
      - SpectralClustering performs a low-dimension embedding of the affinity matrix between samples, followed by clustering, e.g., by KMeans, of the components of the eigenvectors in the low dimensional space.

- ### How to represent graph: types of graphs

  - **Directed vs undirected graphs**

  - **Weighted vs unweighted graphs**

  - **The COO format**
