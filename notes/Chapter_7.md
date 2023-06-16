# Attention and Transformers
1. [Sequence to Sequence Models](#sequence-to-sequence-models)
2. [Attention](#attention)
3. [Key Concepts of Transformers](#key-concepts-of-transformers)
4. [Self-Attention](#self-attention)

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
