import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_charecters, input_size, hidden_size, embedding_dim):
        super().__init__()
        self.n_charecters = n_charecters

        self.char_embedder = torch.nn.Embedding(n_charecters, input_size)
        self.relu1 = torch.nn.ReLU()
        self.token_linear = torch.nn.Linear(embedding_dim, hidden_size)
        self.bilstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.hidden_combine = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.output_combine = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_chars, token_embedding):
        input = self.char_embedder(input_chars)
        h_0 = self.token_linear(token_embedding).unsqueeze(0)
        h_0 = self.relu1(h_0)
        c_0 = torch.zeros_like(h_0)

        # insert hidden layers from both sides of BiLSTM
        h_0, c_0 = torch.cat([h_0, h_0]), torch.cat([c_0, c_0])

        output, (hidden, cell_state) = self.bilstm(input, (h_0, c_0))

        # change back to single dimension (like one-way LSTM)
        output = self.output_combine(output)
        hidden = torch.cat((hidden[0], hidden[1]), dim=-1).unsqueeze(0)
        hidden = self.hidden_combine(hidden)

        return output, (hidden, cell_state)
