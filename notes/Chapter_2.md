# Training Neural Networks
1. [Optimization](#optimization)
2. [Popular Optimization Algorithms](#popular-optimization-algorithms)
3. [Activation Functions](#activation-functions)
4. [Training in PyTorch](#training-in-pytorch)
5. Quiz Yourself on Training Neural Networks

## Optimization
- ### Overview
    - In case of machine learning, optimization refers to minimizing the loss function by systematically updating the network weights.
    - Mathematically:
        $w\'\ = argmin_wC(w)$
        - C: loss function
        - w: weights

- ### Gradient descent

- ### Batch Gradient Descent
    - Impractical for large datasets as computation becomes very expensive.

- ### Stochastic Gradient Descent

- ### Mini-batch Stochastic Gradient Descent
    - In practice, mini-batch stochastic gradient descent is the most frequently used variation because it is both
        - computationally cheap
        - results in more robust convergence

## Popular Optimization Algorithms
- ### Objective
    - Discover the most frequently-used alternatives of gradient descent and the intuition behind them.

- ### Concerns on SGD
    - Limitations:  
        1. If the loss function changes quickly in one direction and slowly in another, it may result in a high oscillation of gradients making the training progress very slow.
        2. If the loss function has a local minimum or a **saddle point**, it is highly likely that SGD will be stuck there without being able to "jump out" and proceed in finding a better minimum.
        3. Noisy gradients update:
            - Since we estimate them based only on a small sample of our dataset.
        4. Tricky to choose a good loss function.
            - Requires time-consuming experimentation with different hyperparameters.
        5. Same learning rate applied to all parameters.
            - Can become problematic for features with different frequencies or significance.

    - Additional info (KA):
        - [Escaping from Saddle Points](https://www.offconvex.org/2016/03/22/saddlepoints/)
            - Critical point
                - Gradient $\nabla f(x) = \hat{0}$
                    - For strongly convex functions, unique critical point which is also the global minimum.
                    - For non-convex functions, gradient to be $\hat{0}$ is not good enough.
                        - Example: 
                            - Function: $y = x_1^2 - x_2^2$
                            - Saddle point: x = (0,0)
                - Types of critical points:
                    - Local min
                    - Local max
                    - Saddle point
                - Second order derivative (usually known as *Hessian*) is considered to distinguish the above types.
            - First order optimization algorithms:
                - Many popular optimization techniques in practice belong to this category.
                - They may get stuck at saddle points:
                    - Since they look at the gradient information, and never explicitly compute the Hessian.
            - In order to optimize non-convex functions with many saddle points, optimization algorithms need to make progress even at (or near) saddle points.
                - Simplest way:
                    - Use second order Taylor's expansion.
                        - If gradient $\nabla f(x)$ is $\hat{0}$, we can still hope to find a vector $u$ where $u^T \nabla^2 f(x) u < 0$.

- ### Adding momentum
    - Borrows the principle of momentum from physics.
    - Enforces SGD to keep moving in the same direction as the previous timesteps.
    - Accomplished using the variables:
        - Velocity $v$
            - Computed as the running mean of gradients up until a point in time.
            - Indicates the direction in which the gradient should keep moving towards.
        - Friction $\rho$
            - Constant number that aims to decay.
    - Weight update:
        - At every time step, velocity is updated by decaying the previous velocity on a factor of $\rho$ and
        - Add the gradients of the weights on the current time.
        - Then weights are updated in the direction of the velocity vector.
    
    - Advantages of using momentum:
        - Can now **escape local minimums** or saddle points.
            - Since we keep moving downwards even though the gradient of the mini-batch might be zero.
        - Can **reduce oscillation of gradients**
            - Since velocity vectors can smooth out the highly changing landscapes.
        - **Reduces moise of the gradients** (stochasticity)
            - Follows a more direct walk down the landscape.

- ### Adaptive learning rate
    - ### Adagrad
        - Keeps a running sum of the squares of the gradients in each dimension.
            - In each update, we scale the learning rate based on square root of the sum.
            - Only magnitude of the gradients is considered and not the sign.
            - Implication:
                - Fast gradient change => Small learning rate
                - Slow gradient change => Big learning rate
        - **Drawback**
            - As time goes by, the learning rate becomes smaller and smaller.
                - Due to monotonic increment of the running squared sum.
    - ### RMSprop
        - "Leaky Adagrad"
            - Add the notion of friction by decaying the sum of the previous squared gradients.
    - ### Adam
        - Adaptive moment estimation
        - Arguably the most popular variation nowadays
        - Combines the two best previous ideas
            - Momentum
            - Adaptive learning rate

## Activation Functions
- ### Sigmoid
    - Advantages
        - Normalizes the output in the range (0,1) so that it can be interpreted as probability.
        - Differentiable
            - Hence can easily run backpropagation.
        - Monotonic
    
    - Disadvantages
        - Tends to flatten at the edges of its range (close to 0 or 1).
            - Gradients become very very small, which poses problems on backpropagation.
        - Output is always positive and not centered around zero.
            - Can leadd to always positive (or negative) gradients.
        - The computation of exponential is quite expensive.

- ### Tanh
    - $f(x) = (e^x - e^{-x})/(e^x + e^{-x})$
    - Hyperbolic function solves some of the above mentioned issues with using sigmoid by:
        - Imposing a range of (-1,1)
        - Centering output around 0
    - Disadvantage:
        - Nullifying gradients close to the edge of its range.

- ### Relu
    - Rectified Linear Unit
    - $f(x) = max(0,x)$
    - Advantages:
        - Does not saturate the gradients.
        - Computationally very efficient.
        - In practice, converges much faster than sigmoid or tanh.

- ### Leaky Relu
    - $f(x) = max(0.01x, x)$
    - ```nn.LeakyReLU(0.01)```
    - Activation won't "die" for negative numbers.

- ### Parametric Relu
    - $f(x) = max(ax,x)$
    - Additional info (KA):
        - Alternative explanation of the equation in [PyTorch's documentation](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
        ```
            PReLU(x) = x, if x >= 0
                     = ax, otherwise
        ```

- ### Softmax
    - Ideal for multi-class problems.
    - IMHO, the following statements led to the confusion in the implementation of the function from scratch:
        - "*We usually apply softmax in the last dimension of a multi-dimensional input. To do that in Pytorch, you can just set ```dim=-1.```*"
            - This corresponds to  
                ```softmax(x) = torch.exp(x)/torch.sum(torch.exp(x), dim=-1).repeat_interleave(x.shape[-1]).reshape(x.shape)```
        - Whereas PyTorch's [Softmax documentation](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) mentions:
            - "*Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and **sum to 1**.*"

## Training in PyTorch
- ### Overview
    - [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
        - Input: 32*32 RGB images
        - Output: 10 classes

- ### Task
    - We will use:
        - Vanilla SGD algorithm
            - ```torch.optim.SGD```
        - Cross-Entropy loss
            - ```torch.nn.CrossEntropyLoss```
- ### How to use the optimizer and gradients
- ### Help
    - Build and train a simple neural network on classifying objects in 10 classes on CIFAR 10 dataset.
        - [Notebook (Question)](../code/train_question.ipynb)
        - [Notebook (Answer)](../code/train_question.ipynb)

    - Additional info (KA):
        - Hint available at PyTorch's [optimization documentation page](https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step).
