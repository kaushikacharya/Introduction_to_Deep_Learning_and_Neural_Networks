# Convolutional Neural Networks
1. [The Principles of the Convolution](#the-principles-of-the-convolution)
2. [Convolution in Practice](#convolution-in-practice)
3. [Build a Convolutional Network](#build-a-convolutional-network)
4. [Batch Normalization and Dropout](#batch-normalization-and-dropout)
5. [Skip Connections](#skip-connections)
6. [CNN Architectures](#cnn-architectures)
7. [Quiz Yourself on CNNs](#quiz-yourself-on-cnns)

## The Principles of the Convolution
- ### Why convolution?
    - The fully connected layer explained in previous chapters doesn't respect the spatial structure of the input.
    - *Inductive bias*:
        - Nearby pixels share similar characteristics and that needs to taken into account by design.
    - Convolutional layers exploit the **local** structure of the data.
    - Convolutional layer is restricted to operate on a local window called kernel.
    - Then we slide this window throughout the input image.

- ### Convolution
    - Given an input matrix N * N and a kernel p * p, where p < N:
        - We slide the filter across every possible position of the input matrix.
        - At each position, we perform a dot product operation and calculate a scalar.
        - We gather all these scalar together to form the output: *feature map*.
    - Intuitively, CNNs are able to recognize patterns in images such as edges, corners, circles etc.
    - Another perspective:
        - CNNs can be thought of as locally connected neural networks, as opposed to fully connected.

- ### Important notes
    - Convolution is a linear operator.
    - The kernel weights are trainable and are shared through the input.
    - Recommended solution for convolution of a 2D image and a 2D kernel.
        ```
        def conv2d(image, kernel):
            H, W = list(image.size())
            M, N = list(kernel.size())

            out= torch.zeros(H-M+1, W-N+1, dtype=torch.float32)
            for i in range(H-M+1):
                for j in range(W-N+1):
                    out[i,j]= torch.sum(image[i:i+M,j:j+N]*kernel)
            return out
        ```
        - Additional info (KA):
            - The answer in [StackOverflow's thread](https://stackoverflow.com/questions/73924697/whats-the-difference-between-torch-mm-torch-matmul-and-torch-mul) explains the difference between:
                - ```torch.mm```
                - ```torch.mul```
                - ```torch.matmul```
            - Since ```torch.mul``` performs elementwise multiplication, we can use it to replace out[i,j] computation in the above code:
                - $out[i,j]= torch.sum(torch.mul(image[i:i+M,j:j+N]*kernel))$

## Convolution in Practice
- ### Overview
    - 3D tensor images
        - Width
        - Height
        - 3 channels (R, G, B)
    - Correspondingly the kernel should be a 3D tensor
        - k * k * channels
    - Kernel produces 2D feature map.
        - The sliding happens only across width and height.
    - In practice, multiple kernels are used to capture different kinds of features at the same time.
    - Learnable weights are the values of the filters and can be trained with backpropagation.
    - In PyTorch, convolutional network is defined similar to
        - ```conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)```
            - 3 channels i.e. R,G,B
            - 5 feature maps (channels)
            - Kernel size: $5*5*3$
                - For simplicity, we just say $5*5$ kernel.

- ### Spatial dimensions
    - Input size: $W_1 * H_1 * D_1$
    - Output channels size: $K$
    - Kernel size: $F * F$
    - Stride: S
    - Padding: P
    - Leads to output of size: $W_2 * H_2 * D_2$
        - $W_2 = (W_1 - F + 2P)/S + 1$
        - $H_2 = (H_1 - F - 2P)/S + 1$
            - ?? As per [PyTorch's Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), instead of $-2P$ it should be $+2P$
        - $D_2 = K$
    - PyTorch code
        - ```conv_layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)```
        - Additional info (KA):
            - [PyTorch's Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) explains the input size:
                - $(N,C_{in},H,W)$
                - N represents batch size

- ### Pooling layer
    - Pooling layers:
        - A way to downsample the features.
        - No learnable parameters.
    - Max-pooling:
        - The most common way.
    - Reasons to introduce pooling:
        - Adds invariance to minor spatial changes.
        - We want to gradually reduce the resolution of the input as we perform the forward pass.
            - The deeper layers should have a higher receptive field i.e.should be more and more sensitive to the entire image.
        - Ultimate goal is to classify an image.
        - Pooling makes the learned features more abstract.
    - PyTorch example code:
        ```
        input_img = torch.rand(1,3,8,8)
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        out = layer(input_img)
        ```

## Build a Convolutional Network
- We will build a fully functional CNN and train it with CIFAR dataset.
- This is an extension of the [previous assignment](./Chapter_2.md#help).
- [Notebook (Question)](../code/cnn_question.ipynb)
- [Notebook (Answer)](../code/cnn_answer.ipynb)
    - Good practice:
        - Use ```relu``` function from [```torch.nn.functional```](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
- Additional info (KA):
    - [Discussion in PyTorch's forum](https://discuss.pytorch.org/t/transition-from-conv2d-to-linear-layer-equations/93850/3) explains the image tensor size at each of the network layers.

## Batch Normalization and Dropout
- ### Batch normalization
    - Related idea: Input scaling
    - Non-normalized input features:
        - Undesirable to train a model with **gradient descent** using these features.

- ### Notations
    - Batch features shape: [N, C, H, W]
        - N: Batch size
        - C: Feature channels
        - H: Height
        - W: Width
    - Batch normalization (BN) normalizes the mean and standard deviation for each individual feature map/channel.
        - Explained y mathematical equation as well as visually.
    - Trainable parameters:
        - $\lambda$
        - $\beta$
    - These trainable parameters result in the linear/affine transformation, which is different for all channels.

- ### Advantages and disadvantages of using batch normalization
    - **Advantages**:
        - Accelerates the training of deep neural networks and tackles the vanishing gradient problem.
        - Beneficial effect on the gradient flow through the network:
            - Reduces the dependence of gradients on
                - Scale of the parameters or
                - Their initial values.
            - This allows us to use higher learning rates.
        - In theory, makes it possible to use saturating nonlinearities by preventing the network from getting stuck.
        - Makes the gradients more predictive.
    - **Disadvantages**:
        - May cause inaccurate estimation of batch statistics when we have a small batch size. This increases the model error.
            - Example: In image segmentation, the batch size is usually too small.
    - Batch Normalization implementation:
        - My Solution
            - [Code](../code/batch_normalization_exercise.py)
            - Not sure why the output is different from the expected output.
            - Broadcasting method idea taken from user kmario23's answer in [StackOverflow thread](https://stackoverflow.com/questions/51097719/add-substract-between-matrix-and-vector-in-pytorch).

        - Official solution:
            - [Code](../code/batch_normalization_official_solution.py)
            - Why standard deviation is computed manually?
                - Shouldn't it be easier if ```torch.std``` is used?

- ### Dropout
    - Conceptually:
        - Dropout approximates training a large number of neural networks with different architectures in parallel.
        - The conceptualization suggests that perhaps dropout breaks-up situations where **network layers co-adapt** to correct mistakes from prior layers, in turn making the model more robust.
    - In practice, during training, some number of layer outputs are randomly ignored (dropped out) with probability p.
    - Connectivity alteration:
        - The same layer will alter its connectivity and will search for **alternate paths** to convey the information in the next layer.
        - As a result, each update to a layer during training is performed with a **different "view"** of the configured layer.
    - "Dropping"
        - **Temporarily removing nodes** from the network for the current forward pass along with its incoming and outgoing connections.
    - Dropout has the effect of making the training process noisy.
    - Dropout increases sparsity of the network and in general encourages sparse representations!

## Skip Connections
- ### The update rule and the vanishing gradient problem

- ### Skip connectione for the win
    - Skip connections skip some layer in the neural network and feed the output of one layer as the input to the next layers, instead of just the next one.
    - Two fundamental ways to skip connections through different non-sequential layers:
        - **Addition**, as in residual architectures.
        - **Concatenation**, as in densely connected architectures.

- ### ResNet: skip connections via addition
    - ResNet: Residual Networks
    - Core idea:
        - To backpropagate through the identity function by just using vector addition.
            - F(x) + x
            - Gradient would then simply be multiplied by one and its value will be maintained in the earlier layers.
    - ResNets stack these skip residual blocks together.
    - Identity function is used to **preserve the gradient**.
    - Another reason for commonly using skip connections:
        - This is apart from the vanishing gradients.
        - Information captured in the initial layers, that we would like to allow the later layers to also learn from them.
            - Tasks such as:
                - semantic segmentation
                - optical flow estimation
    - Learned features in earlier layers correspond to **lower semantic information** that is extracted from the input.
        - Without the skip connection, that information would have turned too abstract.
    - Coding exercise:
        - Implement a skip connection in PyTorch:
        - My solution:
            ```
            class SkipConnection(nn.Module):

                def __init__(self):
                    super(SkipConnection, self).__init__()
                    self.conv_layer1 = nn.Conv2d(3, 6, 2, stride=2, padding=2)
                    self.relu = nn.ReLU(inplace=True)
                    self.conv_layer2 = nn.Conv2d(6, 3, 2, stride=2, padding=2)
                    self.relu2 = nn.ReLU(inplace=True)

                def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
                    # WRITE YOUR CODE HERE
                    h1 = self.conv_layer1(input)
                    self.relu(h1)
                    h2 = self.conv_layer2(h1)
                    self.relu(h2)
                    o = h2 + input
                    return o
            ```

- ### DenseNet: skip connections via concatenation
    - For better understanding, I followed the Aman Arora's blog on [DenseNet and its implementation in TorchVision](https://amaarora.github.io/posts/2020-08-02-densenets.html)
        - In a DenseNet architecture, each layer is connected to every other layer.
        - For each layer, the feature maps of all the preceding layers are used as inputs, and its own feature maps are used as input for each subsequent layers.
            - The input of a layer inside DenseNet is the concatenation of feature maps from previous layers.
        - **Advantages**:
            - Alleviate the vanishing gradient problem.
            - Strenghthen feature propagation.
            - Encourage feature reuse.
            - Substantially reduce the number of parameters.
        - Dividing the network into densely connected blocks:
            - Inside the dense blocks, the feature map size remains the same.
                - This makes the feature concatenation possible.
        - Transition layers:
            - Layers between the dense blocks.
            - It performs convolution + pooling.

## CNN Architectures
- ### AlexNet
    - Trained on [ImageNet](http://www.image-net.org/)
        - Dataset with 1M training images of 1000 classes.

- ### VGG
    - Paper:
        - Very Deep Convoluitonal Networks for Large-Scale Image Recognition.
        - The paper showed evidence that simply adding more layers increases performance.
    - Principles:
        - A stack of three $3*3$ convoluiton layers are similar/even better to a single $7*7$ layer.
            - Reason: Usage of three non-linear activations in between (instead of one) makes the function more discriminative.
        - This design decreases the number of parameters.

- ### InceptionNet/GoogleNet
    - Paper:
        - Going Deeper with Convolutions
    - Motivation:
        - Increasing the depth (number of layers) is not the only way to make a model bigger.
        - How about increasing both the depth and width of the network while keeping computations to a constant level?
        - The inspiration comes from the human visual system, wherein information is processed at multiple scales and then aggregated locally.
            - Challenge: To achieve without a memory explosion.
    - Output padding:
        - Padding *p* and kernel *k* defined so that output spatial dimensions equal input spatial dimensions.
    - Kernel size preference (in general):
        - Larger kernel: Preferred for information that resides globally.
        - Smaller kernel: Preferred for information that is distributed locally.
    - Uses convolutions of different kernel sizes ($5*5, 3*3, 1*1$) to capture details at multiple scales.
    - Computation reduction:
        - $1*1$ convolutions are used to compute reductions before the computationally expensive convolutions ($3*3$ and $5*5$).
        - $1*1$ convolutions work similar to a low dimensional embedding.
    - Addition info (KA):
        - [Visual explanation of kernel dilation](https://www.educative.io/answers/what-is-dilated-convolution)
    
## Quiz Yourself on CNNs
- The axis that we slide the input data defines the dimension of a convolution. For images, it’s a 2D convolution since we “slide” only in the spatial dimensions. But we can still apply convolutions in 1D sequences that have some kind of local structure.
- Skip-connections may provide more paths, however, they tend to make effective receptive field smaller.
    - Source: https://theaisummer.com/receptive-field/
