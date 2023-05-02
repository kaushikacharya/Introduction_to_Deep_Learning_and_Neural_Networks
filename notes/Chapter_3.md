# Convolutional Neural Networks
1. [The Principles of the Convolution](#the-principles-of-the-convolution)
2. [Convolution in Practice](#convolution-in-practice)

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
- This is an extension of the [previous assignment](./Chapter_2.md#training-in-pytorch).
- [Notebook (Question)](../code/cnn_question.ipynb)
- [Notebook (Answer)](../code/cnn_answer.ipynb)
    - Good practice:
        - Use ```relu``` function from [```torch.nn.functional```](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
- Additional info (KA):
    - [Discussion in PyTorch's forum](https://discuss.pytorch.org/t/transition-from-conv2d-to-linear-layer-equations/93850/3) explains the image tensor size at each of the network layers.
