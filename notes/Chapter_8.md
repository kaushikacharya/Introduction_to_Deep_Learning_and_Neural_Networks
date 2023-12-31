# Graph Neural Networks

1. [Basics of Graphs](#basics-of-graphs)
2. [Mathematics for Graphs](#mathematics-for-graphs)
3. [Graph Convolutional Networks](#graph-convolutional-networks)
4. [Implementation of a GCN](#implementation-of-a-gcn)
5. Quiz Yourself on GNNs

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

  - Graph as a composition of structure and signal:
    - Structure = Adjacency Matrix
      - $A \in R^{N \cdot N}$
    - Graph Signal (feature matrix)
      - $X \in R^{N \cdot F}$
      - $F$: Number of features
        - RGB image: $F$ = 3
        - Word: $F$ = embedding dimension

## Mathematics for Graphs

- ### The basic maths for processing graph-structured data

  - Graph signal $X \in R^{N * F}$
  - Adjacency matrix $A \in R^{N * N}$
  - Degree of node:
    - The number of nodes connected to.
    - e.g. non-corner pixel has a degree of 8 (which is the surrounding pixels)
  - Degree matrix $D$:
    - Fundamental in graph theory
      - Provides a single value for each node.
    - Also used for the computation of the most important graph operator: the graph Laplacian.

- ### The graph Laplacian
  
  - Defined as:
    - $L = D - A$
  - Normalized version is used in graph neural networks.
    - To avoid problems when processing with gradient-based methods.
  - $L_{norm} = D^{-1/2} L D^{-1/2}$
    - $L_{norm} = I - D^{-1/2} A D^{-1/2}$
  - A slightly alternated version often used in graph neural networks:
    - $L_{norm} = D^{-1/2} (A + I) D^{-1/2}$
    - From now on, this will be referred as normalized graph laplacian.
    - Additional info (KA):
      - Self-loops are created by adding identity matrix to the adjacency matrix.
        - This ensures that for each node, we sum up all the feature vectors of all neighboring nodes we include that node too.
      - $D^{-1}A$: Corresponds to taking the average of neighboring node features.
      - $D^{-1/2}AD^{-1/2}$: Symmetric normalization
  - With this trick, the input can be fed into a gradient-based algorithm without causing instabilities.
  - Implementation:
    - [Graph Laplacian](../code/graph_laplacian.py)
  - Additional resource:
    - [Thomas Kipf's blog](https://tkipf.github.io/graph-convolutional-networks/)

- ### Laplacian eigenvalues and eigenvectors

  - The multiplicity of the zero eigenvalue of the graph laplacian is equal to the number of connected components.
    - Multiplicity k: k distinct non-trivial eigen vectors.
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

## Graph Convolutional Networks

- ### Objective

  - Discover how Graph Convolutional Networks are conceived.
  - A deep dive into the mathematics behind them.

- ### Types of graph tasks: graph and node classification

  - In [previous lesson](#mathematics-for-graphs) we discussed a bit about the input representation.
  - But what about the target (output)?
  - The most basic tasks in graph neural networks are:
    - **Graph classification**
      - Find a single label for each individual graphs.
        - Similar to image classification.
      - **Inductive learning**
      - Example: Protein type
    - **Node classification**
      - Find a label for the nodes of a graph (usually a huge graph).
      - Formulated as a *semi-supervised* learning task.
        - **Transductive learning**
        - We have very few labeled nodes to train the model.
      - Example: User type in social network.

- ### How are graph convolutions layer formed

  - Principle: Convolution in the vertex domain is equivalent to multiplication in the graph spectral domain.
  - The most straighforward implementation of a graph neural network would be something like:
    - $Y = (AX)W$
      - W: trainable parameter
      - Y: Output
      - A: Usually binary and has no interpretation
  - By definition, multiplying a graph operator by a graph signal will compute a weighted sum of each node's neighborhood.
    - This can be expressed with a simple matrix multiplication.
    - In general, when a graph operator is multiplied by graph signal, transformed graph signal is created.
  - $L_{norm}$: Normalized Laplacian
    - A more expressive operator than $A$.
    - Direct interpretation
  - Locality:
    - $L^K$: Interpreted as expressing the graph constructed from the $K$ hops.
      - Provides the desired localization property.
        - Similar to convolutions.
  - Simplest graph convolution network:
    - $Y = L_{norm}XW$
    - $L_{norm} = D^{-1/2}(A + I)D^{-1/2}$

- ### The background theory of spectral graph convolutional networks

  - The convolution of graph signal $X$ can be defined in the "spectral" domain.
  - Spectral basically means that we will utilize the Laplacian eigenvectors.
  - Therefore, convolution in graphs can be approximated by applying a filter $g$ in the eigenvalues of the Laplacian.
    - $Y = g_\theta (L) X = g_\theta (U \Lambda U^T) X = U g_\theta(\Lambda) U^T X$
      - U: Eigenvectors of $L$
      - $\Lambda$: Diagonal matrix whose elements are the corresponding eigenvalues.

  - **The recurrent Chebyshev expansion**

    - The computation of the eigenvalues would require **Singular Value Decomposition** of $L$.
      - Computationally costly.
    - Hence appromixate with Chebyshev expansion.
    - Similar to approximating a function with a Taylor series based on its derivatives:
      - We compute a sum of its derivatives at a single point.
    - The Chebyshev approximation is a recurrent expansion that estimates $L^K$, avoiding the K square matrix multiplications.
    - The bigger the power, the bigger the local receptive field of the graph neural network will be.
    - We will design a filter $g$ parametrized as a polynomial function of $L$, which can be calculated from a recurrent Chebyshev expansion of order K.
    - To avoid the matrix decomposition, we will work with a rescaled graph Laplacian.
      - $\tilde{L}_h = (2/\lambda_{max})L_{norm} - I_N$
    - Big picture of the graph convolution layer:
      - $Y = g_\theta (\tilde{L}_h) X$
    - Chebyshev expansion in the Laplacian:
      - $\tilde{X}_p = T_p (\tilde{L}_h) X = 2 \tilde{L}_h \tilde{X}_{p-1} - \tilde{X}_{p-2}$
      - Used to avoid doing $K$ multiplications of $L_{norm}$ to calculate the K-th power in the spectral filtering.
      - Index $p$: Indicates the power.
    - In particular, each graph convolution layer will do the following:
      - $Y = g_\theta (\tilde{L}_h) X = [\tilde{X}_0, \tilde{X}_1, ...., \tilde{X}_{K-1}]\theta_v$
        - $\theta_v = [\theta_0, \theta_1, ...., \theta_{K-1}]$ are the learnable coefficients.
    - The first two recurrent terms of the polynomial expansion are calculated as:
      - $\tilde{X}_0 = X$ and
      - $\tilde{X}_1 = \tilde{L}_h X$
    - Thus, the graph signal $X$ is projected onto the Chebyshev basis (powers) $T_p (\tilde{L}_h)$ and concatenated (or summed) for all orders $p \in [0, K-1]$
    - We can imagine the projection onto multiple powers of Laplacian as an inception module in CNNs.
    - As a result, multiple complex relationships between neighboring vertices are gradually captured in each layer.
    - K: Controls the receptive field.
    - $\theta_v$: Trainable parameters
      - Adjust the contribution of each Chebyshev basis.
    - Additional info (KA):
      - [Inception module](https://deepai.org/machine-learning-glossary-and-terms/inception-module):
        - Designed to solve the problem of computational expense, as well as overfitting.
        - Takes multiple kernel filter sizes within the CNN, and rather than stacking them sequentially, orders them to operate on the same level.

  - **What we actually do under the hood: spectral filtering on the eignevalues**
    - Even though an implementation of the pre-described method is provided, we actually raise the eigenvalues to the power K.
    - The power of the Laplacian is applied in the eigenvalues:
      - $L^p = (U \Lambda U^T)^p = U \Lambda^p U^T$
    - Even though we use the laplacian directly, we actually operate on the eigenvalues:  
      - $g_\theta (\Lambda) = \sum_{p=0}^{K-1}\theta_p\Lambda^p = \sum_{p=0}^{K-1} \theta_p T_p (\tilde{\Lambda})$, with
        - $T_p (\tilde{\Lambda}) = 2 \tilde{\Lambda} T_{p-1} (\tilde{\Lambda}) - T_{p-2} (\tilde{\Lambda})$
        - $\tilde{\Lambda} = (2/\lambda_{max}) \Lambda - I$
        - $\theta \in \mathbb{R}^K$: vector of the spectral coefficients.
    - **Insight**:
      - By approximating a higher power K of the Laplacian, we actually design spectral filters that enable each layer to aggregate information from K-hops away neighbors in the graph, similar to increasing the kernel size of a convolutional kernel.

  - #### Illustration of the general graph convolution method

    - [Implementation](../code/chebyshev_approximation_laplacian_powers.py)
      - ?? Why is ```create_graph_laplacian_norm()``` different from $L_{norm}$ defined in this chapter?
      - ```@``` matrix multiplication operator explained in [Ali Sivji's article](https://alysivji.github.io/python-matrix-multiplication-operator.html).
      - ```torch.eig``` has been deprecated since version 1.9 as discussed in [github thread](https://github.com/locuslab/qpth/issues/50).
        - Alternative: ```torch.linalg.eig```
      - Ordering of complex numbers:
        - Complex numbers can not be made into an **ordered field** with the usual addition and multiplication as mentioned in the [StackExchange thread](https://math.stackexchange.com/questions/1032257/ordering-of-the-complex-numbers).
        - Incompatibility of min and max for complex numbers between PyTorch and NumPy discussed in [PyTorch's discussion thread](https://github.com/pytorch/pytorch/issues/36374).
      - [StackExchange thread](https://math.stackexchange.com/questions/67304/do-real-matrices-always-have-real-eigenvalues) discusses that real matrices need not have real eigenvalues.

## Implementation of a GCN

- ### Implementing a 1-hop GCN layer in PyTorch

  - For this tutorial, a simple 1-hop GCN layer will be trained on a small graph dataset [MUTAG](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).
    - The above url contains collected benchmark data sets for the evaluation of graph kernels.

  - **MUTAG dataset characteristics**

    - Each node contains a lebel from 0 to 6.
      - This will be used as a one-hot encoding feature vector.
    - Number of nodes: 188
    - Number of classes: 2

  - Our GCN layer will be defined by the following equations:
    - $Y = L_{norm}XW$
    - $L_{norm} = D^{-1/2}(A + I)D^{-1/2}$
  - The usage of adjacency matrix used in graph convolution layers gives graph neural networks a strong bias to respect the initial graph structure in all their layers.

- ### Practical issues when dealing with graphs

  - Batching graph data
    - Reason: Varying number of nodes
  - For graph classification, how to produce a single label when we have the node embeddings.
    - Readout layer

- ### Assignment

  - [Notebook](../code/gnn.ipynb)
    - torchnet
      - An open-source framework that provides abstractions and boilerplate logic for machine learning.
      - [pypi](https://pypi.org/project/torchnet/)
      - [PyTorch documentation](https://tnt.readthedocs.io/en/latest/index.html)
      - [Paper](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf) published in 2016.
