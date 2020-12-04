from typing import Dict

import pandas as pd
import torch

from lans.constants import TOKEN_STR, SENT_ID, TOKEN_ID
from lans.data_preperation.datasets.base_dataset import BaseDataset


class FasttextDataset(BaseDataset):

    def __init__(self, df: pd.DataFrame, char2label_dict: Dict[str, int], token_embedder, max_word_length: int,
                 max_rows: int = None):
        super().__init__(df, char2label_dict, token_embedder, max_word_length, max_rows)
        self.grouped_df = list(df.groupby(SENT_ID))
        self.max_sent_len = self._max_sent_length(df)

    def __getitem__(self, index):
        sent_id, sent_df = self.grouped_df[index]
        x, y, embeddings = [], [], []
        for token_id, token_df in sent_df.groupby([TOKEN_ID]):
            x.append(self._create_token_chars(token_df))
            y.append(self._create_y(token_df))
            embeddings.append(self._create_token_embedding(token_df))
        x = torch.stack(x)
        y = torch.stack(y)
        embeddings = torch.stack(embeddings)

        return embeddings, x, y

    def __len__(self):
        return len(self.grouped_df)

    def _calculate_token_embeddings(self, df: pd.DataFrame, embedder):
        pass

    def _max_sent_length(self, df):
        return max(df.groupby(SENT_ID).size())

    def _create_token_embedding(self, token_df, **kwargs):
        token = token_df[TOKEN_STR].iloc[0]
        vec = self.token_embedder[token] if token in self.token_embedder else self.token_embedder[':']
        return torch.from_numpy(vec)
