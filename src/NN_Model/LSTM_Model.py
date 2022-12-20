
import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=3, depth=3, device='cuda'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.device = device

        # Build a simple LSTM layer
        self.l1 = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.l2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # output the hidden layers into the desired output shape
        self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)
        self.to(self.device)

    def forward(self, x):
        """
        Given a state input of x, consisting of the most recent state information
        packets, output the predicted next position (a sub-component of the supplied
        state information)

        TODO: we may want to consider predicting both the state and rotation -- not
              necessary, but it allows future predictions if desired.
        """

        batch_size = x.shape[0]

        # initialize tensors for LSTM starting states
        h1 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
        c1 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
        h2 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
        c2 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)

        for input in x.split(1, dim=1):
            h1, c1 = self.l1(torch.squeeze(input, dim=1), (h1, c1))
            h2, c2 = self.l2(h1, (h2, c2))

        out = self.output_layer(h2).unsqueeze(dim=1)
        return out
