# Recurrent Neural Networks
1. [A Simple RNN Cell](#a-simple-rnn-cell)
2. [LSTM: Long Short Term Memory Cells](#lstm-long-short-term-memory-cells)
3. [Writing a Custom LSTM Cell in PyTorch](#writing-a-custom-lstm-cell-in-pytorch)
4. [Connect LSTM Cells Across Time and Space](#connect-lstm-cells-across-time-and-space)
5. Quiz Yourself on RNNs

## A Simple RNN Cell
- ### Overview
    - Recurrent layers are designed for processing sequences.
    - To distinguish RNNs from fully-connected layers, we call the non-recurrent networks feedforward NN.
    - Cell:
        - Basic unit in recurrent networks.
        - Smallest computational unit in recurrent networks.

- ### A minimal recurrent cell: sequence unrolling
    - Sequence Unrolling:
        - A minimal recurrent unit can be created by connecting the current timesteps' output to the input of the next timestep.
            - Core recent principle
    - Preferred unrolling dimension:
        - Time
            - Reason: To learn temporal and often long-term dependencies.
    - By processing the whole sequence timestep by timestep, the algorithm takes into account the previous states of the sequence.

- ### Advantages and intuitions
    - Variable length sequence
        - Majority of common recurrent cells can process sequences of variable length.
    - Shared weights
        - One can view the RNN cell as a common neural network with shared weights for the multiple timesteps.
            - The weights of the cell have access to the previous states of the sequence.
    - How to train?
        - Plain backpropagation will not work with recurrent connections.
        - Backpropagation through time.

- ### Training RNNs: backpropagation through time
    - Input unrolling:
        - Magic of RNN networks
            - Given a sequence $X = [x_1,x_2,...,x_N]$, we process the input timestep by timestep.
    - In essence, backpropagation requires a separate layer for each time step with the same weights for all layers (input unrolling).

- ### Analysis of backpropagation through time
    - PyTorch:
        - ```torch.nn.LSTM()```
        - ```torch.nn.LSTMcell()```
    - One can compute the gradients from multiple paths (timesteps) that are then added to calculate the final gradient.

- ### Limitations
    - Time and space complexity:
        - Asymptotically linear to the input length (timesteps).
    - RNN layers are slow at training
        - Sequence unrolling and backpropagation through time come hand in hand.

- ### An analogy with gradient accumulation
    - A trick for low budget machine learner:
        - Usecase: To train model with a bigger batch size than what our memory supports.
        - Perform a forward pass with the 1st batch and calculate loss without updating the gradients.
        - Repeat for 2nd batch and average the losses from different batches.
        - Example code:
            ```
            accumulate_gradient_steps = 2

            for counter, data in enumerate(dataloader):
                inputs, targets = data
                predictions = model(inputs)
                loss = criterion(predictions, targets)/accumulate_gradient_steps
                loss.backward()
                
                if counter % accumulate_gradient_steps ==0:
                    optimizer.step()
                    optimizer.zero_grad()
            ```
## LSTM: Long Short Term Memory Cells
- ### How does LSTM work?
- ### Notation
    - Dot with outer circle: Element-wise matrix multiplication.
    - $c_t$: Long term memory factor at timestep t
    - $x_t$: Input vector
    - $h_t$: Hidden RNN vector at timestep t

- ### Equations of the LSTM cell
- ### Equation 1: The input gate
    - $i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)$
        - The depicted weight matrices represent the memory of the cell.
        - Input $x_t$: Is in the current input timestamp, while $h$ and $c$ are indexed with the previous timestep.
        - Every matrix W is a linear layer.
    - The dimensionalities of $h$ and $c$:
        - Hidden states parameters in PyTorch's LSTM layer. 

- ### Equation 2: The forget gate
    - $f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)$
    - Almost same as equation 1 but the weight matrices are different.

- ### Equation 3: The new cell/context vector
    - $c_t = f_t \odot c_{t-1} + i_t \odot tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$
    - ### The tanh() part
        - New cell information.
        - We don't want to simply update the cell with the new states.
            - We filter the new cell info by applying an element-wise multiplication with the input gate vector $i$.
    - ### Injecting the forget gate information
        - Instead of just adding the filtered input info, we first perform an element-wise vector multiplication with the **previous** context vector.
            - We would like the model to mimic the forgetting notion of humans as a multiplication filter.

- ### Equation 4: The almost new output
    - $o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)$

- ### Equation 5: The new context
    - $h_t = o_t \odot tanh(c_t)$
    - We want to somehow mix the new context vector $c_t$ (after another activation!) with the calculated output vector $o_t$.
        - This is exactly the point where we claim that LSTMs model **contextual information**.
    - Instead of producing an output as shown in equation 4, we further inject the vector called context. Looking back at equations 1 and 2, one can observe that the previous context was involved.
        - In this way, information based on previous timesteps is involved.
    - This notion of context (long-term memory) enabled the modeling of temporal correlations in long-term sequences.

## Writing a Custom LSTM Cell in PyTorch
- ### Overview
    - LSTM network in PyTorch
        ```
        import torch.nn as nn
        ## input_size -> N in the equations
        ## hidden_size -> H in the equations
        layer = nn.LSTM(input_size= 10, hidden_size=20, num_layers=2)
        ```
        - Number of layers => Number of cells that are connected.
        - This network will have LSTM cells connected together.
        - In this lesson, we will focus on the simple LSTM cell based on the equations.
    - Original proposed equations are described in [previous lesson](#lstm-long-short-term-memory-cells).

- ### Simplication of LSTM equations
    - Modern deep learning frameworks use a slightly simpler version of the LSTM.
        - They disregard $c_{t-1}$ from Equations (1) and (2).
        - Additional info (KA):
            - [PyTorch implementation](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html) also disregards {c_t} from Equation (4).
    - Coding exercise
        - [My answer](../code/custom_lstm_cell_exercise.py)
            - Observation:
                - Adding an additional instance of [Linear class](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (```self.linear_gate_c4```) changes the output of one of the two test cases, even though this instance is not being used.
                    - TODO Find out the reason.
        - [Official solution](../code/custom_lstm_cell_solution.py)
            - Even though ```out_gate``` equation is supposed to consider $c_t$, but here it is not used. The above observation was observed in an attempt to use $c_t$ through ```self.linear_gate_c4```.
                - [Update]: As mentioned above, [PyTorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html) also disregards $c_t$ from the out gate equation.
            - In ```cell_memory_gate``` function, each step is explained.
        - Additional info (KA):
            - [PyTorch options](https://stackoverflow.com/questions/53369667/pytorch-element-wise-product-of-vectors-matrices-tensors) for element-wise product (Hadamard product) of matrices.
                - $A * B$
                - $torch.mul(A,B)$

## Connect LSTM Cells Across Time and Space
- ### Overview
    - Recurrent models are really flexible in the mapping from input to output sequences.
        - Based on the problem we need to modify
            - Input to hidden states
            - Hidden to output states
    - By definition, LSTMs can process arbitrary input timesteps.
    - The output can be tuned by designing which outputs of the last hidden-to-hidden layer are used to compute the desired output.

- ### Code explanation
    - Additional info (KA):
        - [PyTorch's LSTMCell documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)
            - The example shows how to execute LSTMCell.

## Quiz Yourself on RNNs
- The input and forget representations of the gates in an LSTM cell are injected in the equations through:
    - Matrix multiplication
        - ?? Shouldn't this be Element-wise multiplication as per Equation (3)?
