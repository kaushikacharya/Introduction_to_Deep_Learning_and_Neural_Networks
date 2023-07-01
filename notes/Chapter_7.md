# Attention and Transformers

1. [Sequence to Sequence Models](#sequence-to-sequence-models)
2. [Attention](#attention)
3. [Key Concepts of Transformers](#key-concepts-of-transformers)
4. [Self-Attention](#self-attention)
5. [Multi-Head Self-Attention](#multi-head-self-attention)
6. [Transformers Building Blocks](#transformers-building-blocks)
7. [The Transformer's Encoder](#the-transformers-encoder)
8. [The Transformer's Decoder](#the-transformers-decoder)

## Sequence to Sequence Models

- ### Formulating text in terms of machine learning

- ### Seq2Seq

  - Sequence to sequence models
    - Input sequence: $x = (x_1, x_2,...x_n)$ (Sequence of tokens)
    - x -> Encoder -> z (Intermediate representation)
    - z -> Decoder -> $y = (y_1, y_2,....,y_m)$ (Predicted sequence)

- ### A comprehensive view of encoder and decoder

  - Assumption:
    - The **encoder** and **decoder** are stacked RNN/LSTM cells, such as LSTMs.
  - The encoder processed the input and produces one **compact representation**, called $z$, from all the input timesteps.
    - $z$ can be regarded as a compressed format of the input.
  - on the other hand, the decodr receives the context vecotr $z$ and generates the output sequence.

- ### The limitations of RNNs

  - Bottleneck problem:
    - Intermediate repesentation $z$ cannot encode informaiton from all the input timesteps.
  - Vanishing gradient problem:
    - The stacked RNN layers create this well-known problem.

## Attention

- ### Overview

  - Core idea:
    - The context vector $z$ should have access to **all** parts of the input sequence instead of just the last one.
    - In other words, we need to form a **direct connection** with each timestep.
  - Attention:
    - A notion of memory gained from attending at multiple inputs through time.

- ### Attention in the encoder-decoder example

- ### Softmax

- ### Attention in the intermediate representation z

  - Attention is defined as the weighted average of values
    - $z_i = \sum_{j=1}^T \alpha_{ij} h_j$

- ### Attention as an alignment between words

- ### How do we compute attention?

  - Attention $(y_i,h)$:
    - Score between the previous state of the decoder $y_{i-1}$ and the hidden state $h = [h_1,h_2,...,h_n]$
  - Attention score describes the relationship between the two states and captures how "aligned" they are.

- ### Self-attention: the key component of the transformer architecture

  - Self-attention can be thought of as a weighted graph.
  - Effectively, attention is a set of trainable weights that can be tuned using standard backpropagation algorithm.

## Key Concepts of Transformers

- Additional info (KA):
  - [Jay Alammar's blog](https://jalammar.github.io/illustrated-transformer/) on transformers

- ### Overview

  - Sequential treatment of Recurrent Neural Network:
    - With RNN, we treat sequences sequentially to keep the order of the sentence in place.
    - Each RNN component (layer) needs the previous (hidden) output.
    - Stacked LSTM computations are performed sequentially.
  - Self-attention: Fundamental building block of transformer.
    - By simply changing the input representation, we can get rid of sequential processing, recurrency and LSTMs.

- ### Representing the input sentence

  - ### Sets and tokenization
  
    - A simple question that started the transformer revolution:
      - Why don't we feed the entire input sequence so there are no dependencies between hidden states?
      - 1st step: tokenization
      - After tokenization, we project words in a **distributed geometrical space** or simply build word embeddings.

  - ### Word embeddings

    - Words are not discrete symbols.
    - They are strongly correlated with each other.
    - An embedding is a representation of a symbol in a distributed low-dimensional space of **continuous-valued** vectors.
    - Ideally, an embedding captures the semantics of the input by placing semantically similar inputs close together in the embedding space.
  
  - ### Positional encodings

    - When a sequence is converted into a set (tokenization), the notion of order is lost.
    - Sense of order is embedded by slightly altering the embeddings based on the position.
    - Officially, positional encoding is a set of small constants that are added to the word embedding vector before the first self-attention layer.
    - In the transformer paper, the authors came up with the sinusoidal function for the positional encoding.
      - [Figure 2 in Amirhossein Kazemnejad's blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) shows positional encodings as wave lengths.
    
    - ### Feature-based attention: the key, value, and query

      - Key-value-query concepts come from information retrieval systems.
      - Foundation of concept/feature-based lookup:
        - When a particular video is searched (**query**), the search engine maps **query** against a set of **keys** (video, title, description etc.) associated with possible stored videos. Then the algorithm presents the best-matched videos (**values**).
      - Additional info (KA):
        - This concept is explained in Jay Alammar's blog.
          - Under the section **Self-Attention in Detail**, self attention computations are explained in terms of vectors.

    - ### Vector similarity in high dimensional spaces

## Self-Attention

- Enables us to find correlations between different words (tokens) of the input indicating the **syntactic** and **contextual structure** of the sentence.
- $Attention(Q,K,V) = softmax((QK^T)/\sqrt{d_k})V$
  - Scaling factor $\sqrt{d_k}$
    - Makes sure that the vectors won't explode.
- Errata:
  - The lesson mentions $\sqrt{d_k}$ as the number of words in the sentence.
  - IMHO, Jay Alammar's blog mentions $d_k$ as the dimension of the key vectors. (In the transformer paper, $d_k = 64$).

## Multi-Head Self-Attention

- Intuition:
  - Allows us to attend to different parts of the sequence differently each time.
- Practical meaning:
  - The model can better capture **positional information** because each head will attend to different segments of the input. Their combination will give us a more robust representation.
  - Each head will capture different contextual information by correlating words in a unique manner.
- Multi-head attention enables the model to **jointly** attend to information from different input representations (projected in a linear subspace) at different positions.
- Three basic steps:
  - Compute the linear projections into keys, queries, and values.
  - Apply attention to all the projected vectors.
  - Concatenate them and apply a final linear.
- Next come normalization and short skip connections, similar to processing a tensor after convolution or recurrence.
- **TODO**
  - How does training ensures that the eight matrices set learn different representation subspaces?

## Transformers Building Blocks

- ### Short residual skip connections

  - In language, there is a significant notion of a wider understanding of the world and our ability to combine ideas.
  - Humans extensibly utilixe these top-down influences (our expectations) to combine words in different contexts.
  - In a very rough manner, skip connections give a transformer a tiny ability to allow the representations of different levels of processing to interact.

- ### Layer normalization

  - In layer normalization, the mean and variance are computed across channels and spatial dims.
  - Additional info (KA):
    - [Bala Priya's blog](https://www.pinecone.io/learn/batch-layer-normalization/#what-is-layer-normalization)
      - Normalization across features, independently for each sample.
      - Explained with visualization.

## The Transformer's Encoder

- ### Add linear layers to form the encoder

  - The idea of the linear layer after multi-head self-attention is to project the representation in a higher space and then back to the original space.
    - This helps solve stablity issues and counter bad initializations.

- ### Recap: The Transformer encoder

  - Three steps performed to process a sentence:
    - Word embeddings of the input sentence are computed simultaneously.
    - Positional encodings are then applied to each embedding resulting in word vectors that also include positional information.
    - The word vectors are passed to the first encoder block.
  
  - Each block consists of the following layers in the same order:
    - A multi-head self-attention layer to find correlations between each word.
    - A normalization layer.
    - A residual connection around the previous two sublayers.
    - A linear layer.
    - A second normalization layer.
    - A second residual connection.

## The Transformer's Decoder

- ### Overview

  - The decoder consists of all the aforementioned components plus two novel ones. As before:
    - The output sequence is fed in its entirety, and word embeddings are computed.
    - Positional encoding is again applied.
    - The vectors are passed to the first decoder block.
  
  - Each decoder block includes:
    - A **masked** multi-head self-attention layer.
    - A normalization layer followed by  a residual connection.
    - A new multi-head attention layer (known as **encoder-decoder attention**).
    - A second normalization layer and a residual connection.
    - A linear layer and a third residual connection.

  - The decoder block repeats N=6 times.
  - The final output is transformed through a final linear layer.
  - The output probabilities are calculated with the standard softmax function.
    - These probabilities predict the next token in the output sequence.
  - [Figure 1 in Amirhossein Kazemnejad's blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) shows the transformer architecture.

- ### Masked multi-head attention

  - We don't know the whole sentence because it hasn't been produced yet.
  - We mask the next word embeddings by setting them to $-\inf$.

- ### Teacher forcing

  - The mask will change for every new token that we compute.
  - We use the prediction of the model as input.
    - This brings the model closer to the testing setup where we will necessarily have to use the predicted token (word) as input in the next step.

- ### Encoder-decoder attention: where the magic happens
  
  - Encoder-decoder attention is simply the multi-head self attention with the difference being that the query Q  comes from a different source than the keys K and values V.
  - Mentioned as **cross attention** in the literature.
  - This is actually where the decoder processes the final encoded representation.
  - Encoder provides the final keys K and values V.
  - Intuition behind the encoder-decoder attention layer:
    - To combine the input and output sentence.
    - The encoder's output encapsulates the final embedding of the input sentence.
    - We will use the encoder output to produce the key and value matrices.
  - On the other hand, the output of the masked multi-head attention block contains the so far generated new sentence and is represented as the query matrix in the attention layer.
  - The enocder-decoder (cross) attention is trained to associate associate the input sentence with the corresponding output word.
  - The output of the last block of the encoder is used in each decoder block.
