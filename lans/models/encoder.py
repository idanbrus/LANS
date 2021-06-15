import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_charecters, input_size, hidden_size, embedding_dim):
        super().__init__()
        self.n_charecters = n_charecters

        self.char_embedder = torch.nn.Embedding(n_charecters, input_size)
        self.relu1 = torch.nn.ReLU()
        self.token_linear = torch.nn.Linear(embedding_dim, hidden_size)
        self.char_linear = torch.nn.Linear(input_size, input_size)
        self.encoder_lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input_chars, token_embedding):
        input = self.char_embedder(input_chars)
        input = self.char_linear(input)
        h_0 = self.token_linear(token_embedding).unsqueeze(0)
        h_0 = self.relu1(h_0)
        c_0 = torch.zeros_like(h_0)
        output, (hidden, cell_state) = self.encoder_lstm(input, (h_0, c_0))
        return output, (hidden, cell_state)
