# Autoencoders

1. [Generative Learning](#generative-learning)
2. [Basics of Autoencoders](#basics-of-autoencoders)
3. [Variational Autoencoder: Theory](#variational-autoencoder-theory)
4. [Variational Autoencoder: Practice](#variational-autoencoder-practice)
5. Quiz Yourself on Autoencoders

## Generative Learning

- ### Discriminative vs generative models

  - Discriminative models:
    - Learn the probability of a label y based on a data point x.
    - In mathematical terms, denoted by $p(y|x)$.
    - We need to learn a mapping between the data and the classes.
  - Generative models:
    - Learn a probability distribution over the data points without external labels.
    - Mathematically formulated as $p(x)$.
  - Conditional generative models:
    - A category of models that try to learn the probability distribution of the data x conditioned on the labels y.
    - Denoted by $p(x|y)$.
  - Bayes' rule:
    - $p(x|y) = \frac{p(y|x)}{p(y)}*p(x)$
      - Aforementioned model types are somewhat interconnected.
    - This effectively tells us that we can build each type of model as a combination of the other types.

- ### Generative models

  - Probability density function $p(x)$
    - This probability density effectively describes the behavior of our training data and enables us to generate novel data by sampling from the distribution.

- ### Latent variable models

  - Latent variable models aim to model the probability distribution with latent variables.
    - Mathematically, data points $x$ that follow a probability distribution $p(x)$ are mapped into latent variables $z$ that follow a distribution $p(z)$.
  - Latent variables:
    - Transformation of the data points into a **continuous lower-dimensional space**.
  - **Intuitively**, latent variables describe or "explain" the data in a simpler way.
  - Additional info (KA):
    - [What is a latent space](https://stats.stackexchange.com/questions/442352/what-is-a-latent-space)
      - Balraj Ashwath mentions few examples of latent space:
        - Word embedding space
        - Image feature space
          - CNNs in the final layers encode higher-level features in the input image
        - Topic modeling methods
          - Latent Dirichlet Allocation (LDA)
          - Probabilistic Latent Semantic Analysis (PLSA)
        - VAEs & GANs:
          - Aim to obtain a latent space/distribution that closely approximates the real latent space/distribution of the observed data.
          - [MIT's lecture on Deep Generative Models](https://www.youtube.com/watch?v=yFBFl1cLYx8&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=5)
  - We can now define five basic terms:
    - **Prior distribution** $p(z)$
      - Models the behavior of the latent variables.
    - **Likelihood** $p(x|z)$
      - Defines how to map latent variables to the data points.
    - **Joint distribution** $p(x,z) = p(x|z) * p(z)$
      - Multiplication of the likelihood and the prior.
      - Essentially describes our model.
    - **Marginal distribution** $p(x)$
      - Distribution of the original data.
      - Ultimate goal of the model.
    - **Posterior distribution** $p(z|x)$
      - Describes the latent variables that can be produced by a specific data point.
  - Let's define two more terms:
    - **Generation**
      - Refers to the process of computing the data point $x$ from the latent variable $z$.
      - In essence, we move from the latent space to the actual data distribution.
      - Mathematically represented by the likelihood $p(x|z)$.
    - **Inference**
      - Refers to the process of finding the latent variable $z$ from the data point x.
      - Mathematically formulated by the posterior distribution $p(z|x)$.
  - If we assume that we somehow know:
    - Likelihood $p(x|z)$
    - Posterior $p(z|x)$
    - Marginal $p(x)$
    - Prior $p(z)$,
  - we can do the following:
    - [Generation](#generation)
    - [Inference](#inference)

- ### Generation

  - To generate a data point,
    - we can sample $z$ from $p(z)$,
      - $z \sim p(z)$
    - and then sample the data point $x$ from $p(x|z)$
      - $x \sim p(x|z)$

- ### Inference

  - To infer a latent variable,
    - we sample $x$ from $p(x)$,
      - $x \sim p(x)$
    - and then sample $z$ from $p(z|x)$
      - $z \sim p(z|x)$
  
  - Fundamental question of latent variable models:
    - How all those distributions can be found?
      - Answer: Variational Autoencoders (VAE)

## Basics of Autoencoders

- Autoencoders are simple neural networks such that their output is their input.
- Goal:
  - Learn how to reconstruct the input data.
- Additional info (KA):
  - [PyimageSearch article on Introduction to Autoencoders](https://pyimagesearch.com/2023/07/10/introduction-to-autoencoders/#h3What)
    - Figure 2 describes the architecture of Autoencoder
  - [Variational Autoencoders: A Vanilla Implementation](https://mlarchive.com/deep-learning/variational-autoencoders-a-vanilla-implementation/)
    - Main goal:
      - To generate high quality output data (e.g. images, texts, or sounds) that belong to the same distribution of the input data.
    - Three main families of generative models:
      - Variational autoencoders (VAE)
      - Generative Adversarial Network (GANs)
      - Diffusion Models
- 1st part:
  - Encoder
    - Receives the input and encodes it in a latent space of a lower dimension (the latent variables $z$).
      - *Input* $=>$ *Encoder* $=>$ *Latent Space*
    - We can think of the latent space as a continuous low-dimensional space.
      - ?? continuous: Does it refers to dense vector representation?
- 2nd part:
  - Decoder
    - Takes the low-dimensional vector and decodes it in order to produce the original input.
      - *Latent Space* $=>$ *Decoder* $=>$ *Output*
- Applications of the latent vector $z$ includes:
  - Compression
    - The latent vector $z$ is a compressed representation of the input.
  - Dimensionality reduction
- We can apply them to entirely novel data. Practical applications include:
  - Data denoising:
    - Feed the network with a noisy image and train them to output the same image but without the noise.
  - Training data augmentation
  - Anomaly detection:
    - Train an autoencoder on data from a single category so that every anomaly gives a large reconstruction error.
- Vanilla autoencoders are trained using a reconstruction loss, which in its simplest form is nothing more than the L2 distance.

- ### Exercise

  - Description of the model i.e. each of the layers is written as comment in the [code](../code/autoencoder_exercise.py).
  - Few things to notice:
    - In the first part of the network, the size of the input is gradually decreasing, resulting in a compact latent representation.
    - In the second part, [ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) layers are increasing the size with the goal to output the original size on the final layer.
      - Additional info (KA):
        - [Transposed convolution animation](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#transposed-convolution-animations)
    - Each layer is followed by a ```ReLu``` activation.
    - The final output is passed through a ```Sigmoid```.
  - Additional info (KA):
    - Calculate number of CNN parameters:
      - Explained by the user *hbaderts* in [StackOverflow thread](https://stackoverflow.com/questions/42786717/how-to-calculate-the-number-of-parameters-for-convolutional-neural-network/42787467)
      - Weights: $n*m*k*l$
        - $n * m$: filter size
        - $l$: Number of feature maps as input
        - $k$: Number of feature maps as output
      - Bias: $k$
        - Bias term for each output feature map.
      - Hence total number of parameters:
        - $(n*m*l + 1)*k$
    - Calculation for the exercise:
      - 1st CNN layer:
        - Number of parameters = $(F*F*l + 1)*k$
          - $(F*F*3 + 1)*12 = 588$
            - Kernel size: $F*F$
            - Solving above equation gives the kernel size: $F = 4$
        - Next we need to calculate the parameter values based on the equation defined in [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
          - $H_{out} = (H_{in} + 2P - D(F-1) - 1)/S + 1$
            - $P$: Padding
            - $D$: Dilation
            - $S$: Stride
          - One possible solution:
            - $P = 1$
            - $D = 1 (default)$
            - $S = 2$
      - Similarly parameter values for rest of the CNN layers can be computed.
  - Solution:
    - [Kaushik](../code/autoencoder_exercise.py)
      - Learning:
        - ```ReLU``` layers needs to be explicitly defined as the test cases in the course checks for the network layers.
          - Simply executing ```torch.relu``` inside ```forward``` function throws error for the test cases.
    - [Official Solution](../code/autoencoder_official_solution.py)
      - Good practice:
        - Encoder and Decoder layers are grouped together using ```nn.Sequential```.

## Variational Autoencoder: Theory

- ### Overview

  - In simple terms, a variational encoder is a probabilistic version of autoencoders.
  - Each latent variable $z$ that is generated from the input represents a probability distribution (posterior distribution denoted by $p(z|x)$).
  - Encoder approximates the posterior by computing another distribution $q(z|x)$, known as the **variational posterior**.
  - A probability distribution is fully characterized by its parameters.
    - So it is enough to pass the parameters of the probability distribution in the decoder instead of simply passing the latent vector $z$ like the simple autoencoder.
  - The decoder receives the distribution parameters and tries to reconstruct the input x.
  - Challenge:
    - One cannot backpropagate through a sampling operation.
  - Additional resources (KA):
    - [Matthew Bernstein's blog on Variational autoencoders](https://mbernste.github.io/posts/vae/)
      - Author explains VAE and related concepts in detail with mathematical equations.
      - Two complimentary ways of viewing the VAE:
        - As a probabilistic model that is fit using variational Bayesian inference.
        - As a type of autoencoding neural network.
    - [PyimageSearch article on Introduction to Autoencoders](https://pyimagesearch.com/2023/07/10/introduction-to-autoencoders/)
      - Both encoder and decoder can be fully connected, convolutional or recurrent, depending on the input data's nature and the autoencoder's architecture.
      - Various types of autoencoders are described.
      - Applications of autoencoders explained.
      - Difference with GANs explained.

- ### Train a variational autoencoder

  - True posterior: $p(z|x)$
  - Variational posterior: $q(z|x)$
    - The course mentions variational posterior as $p(z|x)$ which in my opinion is incorrect as per the statement mentioned few sentences above.
  - Goal:
    - Variational posterior as close as possible to the true posterior.
  - Evidence Lower Bound (ELBO):
    - Additional resources (KA):
      - [Matthew Bernstein's blog on ELBO](https://mbernste.github.io/posts/elbo/)
        - Explains definition, context and derivation.
      - [Matthew Bernstein's blog on Variational Inference](https://mbernste.github.io/posts/variational_inference/)
        - Explains mathematically how variational inference minimizes KL-divergence by maximizing a surrogate quantity ELBO.
      - [Faculty of Khan's introduction to Variational Calculus](https://www.youtube.com/watch?v=6HeQc7CSkZs)
        - Explained with two examples:
          - Minimum path between two points.
          - Minimum time taken to travel from one point to another where velocity is dependent on position.
    - Loss = Reconstruction term - KL Divergence
      - 1st term: Reconstruction term
        - Controls how well the VAE reconstructs a data point $x$ from a sample $z$ of the variational posterior.
        - Known as **negative reconstruction error**.
        - If data points are binary (follow the Bernoulli distribution):
          - reconstruction term can be proved to:
            - $log\, p_\theta(x_i|z_i) = \sum_{j=1}^n[x_{ij}log\,p_{ij} + (1 - x_{ij})log(1-p_{ij})]$
      - 2nd term: KL Divergence:
        - Controls how close the variational posterior is to the prior.
        - If we assume Gaussian prior distribution:
          - -(1/2)$\sum_{j=1}^J(1 + log(\sigma_j^2) -\mu_j^2 - \sigma_j^2)$
  
- ### ELBO Implementation

  - In practice, we used closed analytical forms to compute the ELBO.
  - Reconstruction term:
    - When the data points are binary (follow the Bernoulli distribution), the equation is simply the binary cross entropy.
      - Implemented in PyTorch using ```torch.nn.BCELoss(reduction='sum')```
  - KL-Divergence term:
    - If we assume that the prior distribution is a Gaussian, then KL-Divergence also has a closed form.
    - Additional resource (KA):
      - [Derivation of KL-divergence term when the variational posterior and prior are Gaussian](https://mbernste.github.io/posts/vae/)
        - Appendix section of Matthew Bernstein's blog
      - [Derivation of KL Divergence for Gaussian distribution](https://leenashekhar.github.io/2019-01-30-KL-Divergence/)
        - [Corrected Cross-Entropy derivation](https://github.com/LeenaShekhar/leenashekhar.github.io/pull/1)
  - Solution:
    - [Kaushik](../code/elbo_exercise.py)
    - [Official Solution](../code/elbo_official_solution.py)

- ### Reparameterization trick

  - Intuition:
    - Because we cannot compute the gradient of an expectation, we want to rewrite the expectation so that the distribution is independent of the parameter $\theta$.
      - [Matthew Bernstein's blog on reparameterization trick](https://mbernste.github.io/posts/reparameterization_vi/) mentions that the challenge with the computing gradient of expectation is the integral.
  - Formulating the abstract idea:
    - Transform a sample from a **fixed**, known distribution to a sample from $q_{\phi}(z)$.
    - If we consider the Gaussian distribution, we can express $z$ wrt. a fixed $\epsilon$, where $\epsilon$ follows the normal distribution $N(0,1)$.
    - So now $\epsilon$ is the stochastic term.
  - Backpropagation:
    - Since backpropagation cannot be performed in a fully stochastic operation, hence we will not backpropagate through $\epsilon$.
    - So, instead:
      - A fixed part stochastic with $\epsilon$ is kept.
      - The mean and the standard deviation are trained.
    - Next, backpropagate through the mean $\mu$ and the standard deviation $\sigma$ that are outputs of the encoder.
    - Define the latent space vector $z$ as
      - $z = \mu + \sigma\epsilon$ with $\epsilon$ ~ $N(0,1)$
    - $\epsilon$ term introduces the stochastic part and is **not** involved in the training process.
    - Therefore, we can now compute the gradient and run backpropagation of ELBO w.r.t. variational parameters $\theta$.
  
- ### Reparameterization Trick Exercise

  - Implementation of reparameterization trick in Pytorch
  - Solution:
    - [Kaushik](../code/reparameterization_trick_exercise.py)
    - [Official Solution](../code/reparameterization_trick_official_soluiton.py)
      - IMHO, sampling equation is incorrect.
        - $sample = mu + (eps * var)$
          - Instead of $var$ we should use $std$.
        - Update: My doubt is valid as seen in the code snippet provided in the next lesson.
      - Refer [mlarchive's implementation](https://mlarchive.com/deep-learning/variational-autoencoders-a-vanilla-implementation/)
        - Have a look at the section **Reparameterization Trick**

## Variational Autoencoder: Practice

- ### Overview
    - [Variation Autoencoder](../code/vae.py) code
        - Implemented using 2 linear layers for both the encoder and decoder.

- ### Reparameterization trick
    - In order to generate samples from the encoder and pass them to the decoder, we also need to utilize the reparameterization trick.
    - Remember: We need to be able to run the backward pass during training.
    - The two networks are trained jointly by maximizing the ELBO.
        - VAE case:
            - $L_{\theta,\phi}(x) = E_{q_\phi(z|x)}[log\,p_\theta(x|z)] - KL(q_\phi(z|x)||p_\theta(z))$ 

- ### Analysis of loss terms
    - Reconstruction loss
        - The weird expectation term
        - This loss is just the binary cross-entropy between the latent variable and the input.

- ### Training loop code in PyTorch

- ### Code analysis
    - During training:
        - A data point is passed to the encoder, which outputs
            - mean
            - log-variance of the approximate posterior.
        - Reparameterization trick applied.
        - Pass the reparameterized samples to the decoder to output the likelihood.
        - Compute the ELBO and backpropagate the gradients.
    - Generation of a new data point:
        - Sample a set of latent vectors from the normal prior distribution.
        - Obtain the latent variables from the encoder.
        - Decoder transforms the latent variable of the sample to a new data point.

- ### Homework task
    - Task:
        - Define a Convolutional VAE.
        - Load the CIFAR10 data.
        - Train the VAE.
    - Additional resources (KA):
        - [CIFAR tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py)
            - Shows usage of [torchvision.datasets.CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) to extract/download dataset.
                - Enables code execution locally.
    - Learnings (KA):
        - Incorrect usage of ```BCELoss``` can lead to negative loss:
            - https://discuss.pytorch.org/t/bce-loss-giving-negative-values/62309/3
            - [BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) documentation says:
                - *measures the Binary Cross Entropy between the target and the input probabilities*
                - *Note that the targets $y$ should be numbers between 0 and 1*.
            - Case: Usage of [Template code](../code/vae.py) provided in this lesson led to above situation as the transformed image pixel values were beyond the accepted range of ```target```. It had negative values (observed minimum: -1).
        - Solution:
            - [Matthew Bernstein's blog](https://mbernste.github.io/posts/vae/) suggests *mean squared error*:
                - In the section *Viewing the VAE loss function as regularized reconstruction loss*, author shows that analytical form of $log\, p_\theta(x_i|z_i)$ contains the squared error of simple autoencoder.
                - **Doubts**:
                    - Why should $\sigma_{decoder}$ be ignored in the loss function?
                    - Isn't [GaussianNLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss) a more suitable loss function than [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)?
                    - Or should we have a custom loss function as shown in [machinelearningmastery's](https://machinelearningmastery.com/loss-functions-in-pytorch-models/) section on *custom loss function*.
                    - What should be the analytical form of the *expectation*?
                        - The Educative course mentions that for binary data points the analytical form can be represented by [BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html).
    - Solution:
        - [Kaushik](../code/vae.ipynb)
            - Observation:
                - Reconstructed images are quite blurred.
