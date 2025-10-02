import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN model. Defines LSTMCell and fully connected layer.

        Args:
        input_size: Dimension of the input features.
        hidden_size: Dimension of the hidden state in the LSTM.
        output_size: Dimension of the output layer.
        """

        super().__init__()
        # Define the parameters of the RNN 
        self.hidden_size = hidden_size

        # TODO: Uncomment the next two lines and replace the ???s
        # self.lstm = nn.LSTMCell(???, ???)
        # self.fc = nn.Linear(???, ???)
        
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of LSTMCell and fully connected layer.
        Biases are initialized to zero and weights using Xavier uniform initialization.
        """
        # TODO: Initialize the weights of the RNN

        # NOTE:: Use "for name, param in self.lstm.named_parameters():" to 
        # loop over the parameters of the LSTMCell. For the fully connected 
        # layer, use xavier_uniform_ to initialize the weights and constant_ to 
        # initialize the bias to zero.



    def forward(self, x):
        """
        Forward pass of the model. Processes the input sequence through the LSTM
        and returns the final output after applying sigmoid.

        Args:
        x: Input tensor of shape (N, T, d), where N is batch size,
           T is sequence length, and d is input feature dimension.

        Returns:
        Output tensor after processing the sequence through LSTM and the fully connected layer.
        """
        N, T, d = x.shape

        # Initialize the hidden state and cell state
        h_t, c_t = self.init_hidden(x.size(0))
        
        # TODO: Define the forward pass of the RNN 

        # NOTE: You need to loop over the time steps and update the hidden 
        # state. After the loop, you need to apply the fully connected layer to 
        # the hidden state and apply the sigmoid activation function


        z = torch.tensor(0) # Replace with your implementation 


        return z

    def init_hidden(self, N):
        """
        Initializes the hidden state and cell state for LSTM with zeros.

        Args:
        N: Batch size

        Returns:
        A tuple of (hidden state, cell state), both initialized to zeros with shape (N, hidden_size).
        """

        # TODO :Initialize the hidden state and cell state

        hidden = (0,0) # Replace with your implementation


        return hidden
