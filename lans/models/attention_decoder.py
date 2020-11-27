import torch
from torch import nn
import torch.nn.functional as F



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.3):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout2 = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, cell_state, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout1(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = self.dropout2(output)

        output = F.relu(output)
        output, (hidden, cell_state) = self.lstm(output, (hidden, cell_state))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, cell_state, attn_weights