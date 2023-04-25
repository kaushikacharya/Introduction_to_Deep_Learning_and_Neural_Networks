# Neural Networks
1. [Linear Classifiers](#linear-classifiers)
2. [Optimization and Gradient Descent](#optimization-and-gradient-descent)
3. [Neural Networks](#neural-networks)
4. [Backpropagation Algorithm](#backpropagation-algorithm)
5. [Build a Neural Network with PyTorch](#build-a-neural-network)
6. Quiz Yourself on Neural Networks

## Linear Classifiers
- ### Goal:
    - Explore linear classifiers, their principles, and their training process.
- ### What is a linear classifier?
    - Additional info (KA):
        - [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
            - Applies a linear transformation to the incoming data
                - $y = A^Tx + b$
        - [Visualization](https://www.sharetechnote.com/html/Python_PyTorch_nn_Linear_01.html)
            - ```torch.nn.Linear(n,m)```
                - This module creates single feed forward network with n inputs and m outputs.

- ### Training a classifier

- ### Loss function
    - We will use the mean squared error distance.

- ### Optimization and training process

## Optimization and Gradient Descent
- ### Overview
    - Loss function:
        - In 2D example, it can be thought of as a parabolic-shaped function that reaches its minimum on a certain pair of $w_1$ and $w_2$.
        - Actual shape of the loss unknown.
        - But we can calculate the slope in a point and then move towards the downhill direction.

- ### Slope: the derivative of the loss function
- ### Computing the gradient of a loss function
- ### Summing up the training scheme

## Neural Networks
- ### What is a neuron?
    - Non-linear functions between linear layers enables modeling complex representations with less linear layers.
        - Non-linearities makes neural networks rich function approximators.

- ### Multilayer perceptron

- ### Universal approximation theorem
    - According to universal approximation theorem, given enough neurons and the correct set of weights, a multi-layer neural network can approximate any function.
    - Neural networks are also very good feature extractors.

- ### Deep neural networks as feature extractors
    - Feature extraction:
        - Transformation of the input data points from the input space to the feature space where classification is much easier.
    - In most real-life applications:
        - Only the last one or two layers of a neural network performs the actual classification.
        - The rest account for feature extraction and learning representation.

## Backpropagation Algorithm
- ### Overview:
    - Neural networks:
        - Non-linear classifiers that can be formulated as a series of matrix multiplications.
        - Difficulty arises in computing the gradients.

- ### Notations
- ### Forward pass
- ### Backward pass
    - The error is propagated backwards.

- ### The chain rule for the backward pass

## Build a Neural Network with PyTorch
- ### PyTorch basics
    - [PyTorch](https://pytorch.org/)
        - An open-source Python deep learning framework.
        - Enables us to build and train neural networks.
    - **Tensor**
        - Fundamental building block of PyTorch.
        - N-dimensional array.

- ### Build a neural network
    - An example of developing a neural network using a sequential order of individual layers:
        ```
        nn.Sequential( 
            nn.Linear(2, 3), 
            nn.Sigmoid(), 
            nn.Linear(3, 2), 
            nn.Sigmoid() 
        )
        ```

- ### Program your own neural network
    - Alternative way to define the above neural network:
        ```
        class Model(nn.Module): 
        def __init__(self): 
            super(Model, self).__init__() 
            self.linear1 = nn.Linear(2, 3) 
            self.linear2 = nn.Linear(3, 2) 
    
        def forward(self, x): 
            h = torch.sigmoid(self.linear1(x)) 
            o = torch.sigmoid(self.linear2(h)) 
            return o 
        ```

    - To run forward propagation:
        ```
        model = Model()
        y = model(x)
        ```