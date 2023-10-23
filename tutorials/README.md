# Tutorials

## About

- This section contains the tutorials from external sources.

## Contents

### Quantization

- [Notebook](../code/quantization.ipynb)
  - Task: To classifiy MNIST digits with a simple [LeNet architecture](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html).
  - [Source](https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d)
    - TODO Figure out why ```in_features``` of 1st fully connected layer is 256 and not 400 as shown in LeNet architecture tutorial.
  - Observation:
    - Additinal step in the [PyTorch tutorial on quantization](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training):
      - Batch norm mean and variance estimates were also frozen after few epoch in the train.
- Recommendation:
  - [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)
- PyTorch documentation:
  - [Fuse modules](https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization): combine operations/modules into a single module to obtain higher accuracy and performance.
- [Introduction to Quantization on Pytorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
  - Three modes of quantization supported in PyTorch
    - [Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)
      - Converting from floating point to integer values
        - Mutiplying the floating point value by some **scale factor** and rounding the result to a whole number.
      - The various quantization approaches differ in the way they approach determing that **scale factor**.
      - Key idea with dynamic quantization:
        - Determine the scale factor for activations dynamically based on the data range observed at runtime.
      - Benefits
        - Reduction of model size.
        - Latency reduction due to combination of effects that include
          - Less time spent moving parameter data in
          - Faster INT8 operations
    - Post-training static quantization
    - Quantization aware training
      - Typically results in highest accuracy among these three types.
      - All weights and activations are "fake quantized" during both the forward and backward passes of training.
        - Float values are rounded to mimic int8 values, bit all computations are still done with floating point numbers.
