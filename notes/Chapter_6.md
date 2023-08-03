# Generative Adversarial Networks

1. [Generator and Discriminator](#generator-and-discriminator)
2. [Generative Adversarial Networks in Detail](#generative-adversarial-networks-in-detail)
3. [Develop a GAN with PyTorch](#develop-a-gan-with-pytorch)
4. Quiz Yourself on GANs

## Generator and Discriminator

- ### Adversarial attacks

  - Construction of adversarial example:
    - Visually indistinguishable image
    - But classified incorrectly by the trained classifier model.
  - Adversarial training:
    - Common approach to address the problem:
      - Inject adversarial examples into the training set.
    - Increases the neural network's robustness.
    - Example generation methods:
      - Adding noise
      - Applying data augmentation techniques
      - Perturbating the image in the opposite direction of the gradient (to maximize loss instead of minimizing it).

- ### An example

    ```python
    import torch
    ## FGSM attack code
    def fgsm_attack(image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image
    ```

- ### From adversarial attacks to generative learning

  - In generative adversarial learning, we focus on producing representative examples of the dataset instead of making the network more robust to perturbations.
  - Generator:
    - Another network that does the above work.

- ### Generator

  - **Input**

    - Simplest form:
      - The input is random noise that is sampled from a probability distribution in a small range of real numbers.
        - Called **latent** or **continuous space**.
    - **Stochasticity**:
      - Every time the sample is from a random distribution, a different vector sample will be received.
      - Reason for expected value being used in the GAN papers.

  - **Output**

    - We focus on generating representative examples of a specific distribution (e.g. dogs, paintings etc.)

- ### Discriminator

  - The discriminator is simply a classifier.
  - Instead of classifying an image in the correct class, **focus is on learning the distribution of the class**.
  - Desired class is known.
    - We want the classifier to quantify how representative the class is to the real class distribution.
    - Discriminators output a single probability:
      - 0: Corresponds to the fake generated images.
      - 1: Corresponds to the real samples from our distribution.
  - In game theory, this adversary is called a 2-player min-max game.
    - The generator G attempts to produce fake examples that are close to the real distribution so as to fool discriminator D, while D tries to decide the origin of the distribution.
  - Indirect training:
    - Indirect: We do not minimize the pixel-wise euclidean distance.
      - The gradients are simply computed from the binary classification of the discriminator.
  - Each model is trained in an alternating fashion.

## Generative Adversarial Networks in Detail

- ### Overview

  - Another way to explain GANs is through the probabilistic formulation used in variational autoencoders.
  - Instead of computing $p_{data}(x)$, we only care about the ability to sample data from the distribution.
  - Latent variable $z$:
    - Prior distribution: $p(z)$
      - Usually a simple random distribution such as uniform or a Gaussian distribution.
    - Sample $z$ from $p(z)$ and pass the sample to the generator network $G(z)$.
    - Output a sample of data $x$ with $x = G(z)$.
  - $x$ can be thought of as a sample from the generator's distribution $p_G$.
    - The generator is trained to convert random $z$ into fake data $x$.
      - In other words, force $p_G$ to be as close as possible to $p_{data}(x)$.

- ### Training

  - A key insight: indirect training
  - What it means?
    - The generator is not trained to minimize the distance to a specific image, but just to fool the discriminator!
  - Adversarial loss:
    - Enables the model to learn in an unsupervised manner.
  - Training process of the generator:
    - In this stage, discriminator is not trained.
  - Training process of the discriminator:
    - In this stage, generator is not trained.

- ### The math

  - The discriminator performs gradient ascend while the generator performs gradient descent.
  - Discriminator needs to access both real and fake data while generator has no access to real images.

## Develop a GAN with PyTorch

- Additional info (KA):
  - [NCHW format](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html)

- ### Generator

  - **Input**: 100-length noise vector $z$ sampled from a distribution.
  - **Output**: Flattened image in the dimensions of the cifar10 dataset: $32*32*3 = 3072$

- ### Discriminator

  - **Input**: Output of the generator.
  - **Output**: A scalar in [0,1]
  - **Loss**: Binary cross entropy loss computed based on the output of the discriminator and the data label.

- ### Train the discriminator

  - For the discriminator, we have:
      1. Run a forward and backward pass on real training data.
      2. Run a forward and backward pass on fake training data.
      3. Update the weights.
  - Additional info (KA):
    - [torch.autograd.Variable](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)
      - Variable API has been deprecated.
      - Still work as expected, but they return Tensors instead of Variables.
      - One can create tensors with ```requires_grad=True``` using factory methods.
      - Original purpose of Variables was to be able to use automatic differentiation. Refer Florian Blume's answer in the [stack overflow thread](https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable).

- ### Train the generator

  - For the generator:  
      1. Generate a noise vector.
      2. Perform another pass on the discriminator with the fake data.
      3. Calculate the generator's loss and update the weights.

- [Notebook](../code/gans.ipynb)
  - Task: Build and Train GAN
  - Observation:
    - Initially the discriminator probability for real images are high and low for fake generated images.
        As training progresses, the discriminator probabilities for both classes tends towards 0.5
    - *At convergence, the generator's samples are indistinguishable from real data, and the discriminator outputs $\frac{1}{2}$ everywhere.*
      - Source: [Deep Learning by Goodfellow et al.](https://www.deeplearningbook.org/) (Section 20.10.4 Generative Adversarial Networks)
  - Additional info (KA):
    - Multiple loss combination is explained in a [StackOverflow thread](https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch).
def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
      - Usage in ```train_discriminator```.
    - Generator produces values in the range (-1,1) due to tanh in the final layer.
      - ```log_images``` calls ```vutils.make_grid``` where parameter ```normalize``` normalizes it in the range (0,1).
  - TODO:
    - Experiment with GANs by adding a supervised pixel-wise loss (```torch.nn.L1Loss```) in the generator's output.
