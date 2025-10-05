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
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

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

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0.0)

            



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

        for t in range(T):
            h_t, c_t = self.lstm(x[:, t, :], (h_t, c_t)) # use previous state

        # map final hidden to output
        z = self.fc(h_t)

        z = torch.sigmoid(z)

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

        hidden = torch.zeros(N, self.hidden_size)
        cell = torch.zeros(N, self.hidden_size)

        return (hidden, cell)
