import abc
import logging
from typing import Dict

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from lans.config import JOINT_LABELS_COL
from lans.constants import TOKEN_STR, SENT_ID, TOKEN_ID, FORM, EOW, NULL_LABEL
import torch


class BaseJointDataset(Dataset):

    def __init__(self, df: pd.DataFrame, char2label_dict: Dict[str, int], pos_encoder: Dict[str, int], token_embedder,
                 max_word_length: int, max_rows: int = None, ):
        self.pos_encoder = pos_encoder
        df = df.head(max_rows) if max_rows is not None else df  # just for testing purposes

        self.token_embeddings = self._calculate_token_embeddings(df, token_embedder)
        self.group_tokens = list(df.groupby([SENT_ID, TOKEN_ID]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.token_embedder = token_embedder
        self.char2label_dict = char2label_dict

        self.max_word_length = max_word_length


    def __getitem__(self, index):
        (sent_id, token_id), token_df = self.group_tokens[index]

        token_embedding = self._create_token_embedding(sent_id, token_id)
        token_chars = self._create_token_chars(token_df)
        y, pos_tags = self._create_y(token_df)

        return token_embedding, token_chars, y, pos_tags

    def __len__(self):
        return len(self.group_tokens)

    @abc.abstractmethod
    def _calculate_token_embeddings(self, df: pd.DataFrame, embedder):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_token_embedding(self, sent_id, token_id):
        raise NotImplementedError

    def _create_token_chars(self, token_df):
        chars = [self.char2label_dict[char] if char in self.char2label_dict else 0 for char in
                 token_df[TOKEN_STR].iloc[0]]
        token_chars = torch.ones(self.max_word_length, dtype=torch.long) * self.char2label_dict[EOW]
        token_chars[:len(chars)] = torch.tensor(chars, dtype=torch.long)
        return token_chars

    def _create_y(self, token_df):
        morphemes = ' '.join(token_df[FORM].dropna())
        labels = [self.char2label_dict[char] if char in self.char2label_dict else 0 for char in morphemes]
        labels.append(self.char2label_dict[' '])

        chars, tags = [], []
        for i, row in token_df.dropna().iterrows():
            # pos_tag = self.pos_encoder[row[POS]] if row[POS] in self.pos_encoder else self.pos_encoder[NULL_POS]
            # tags += ([pos_tag] * (len(row[FORM]) + 1))

            tags += [self.pos_encoder[NULL_LABEL]] * len(row[FORM])
            tags.append(self.pos_encoder[row[JOINT_LABELS_COL]] if row[JOINT_LABELS_COL] in self.pos_encoder else
                        self.pos_encoder[NULL_LABEL])

        y = torch.ones(self.max_word_length, dtype=torch.long) * self.char2label_dict[EOW]

        y[:len(labels)] = torch.tensor(labels, dtype=torch.long)

        pos_tags = torch.zeros(self.max_word_length, dtype=torch.long)
        pos_tags[:len(labels)] = torch.tensor(tags, dtype=torch.long)

        return y, pos_tags

    @staticmethod
    def find_max_word_length(df) -> int:
        max_length = 0
        for (sent_id, token_id), token_df in tqdm(df.groupby([SENT_ID, TOKEN_ID]), desc='Calculating max word length..',
                                                  unit='wordb'):
            word_lengths = token_df[FORM].apply(lambda token: len(str(token)))
            token_len = word_lengths.sum() + len(word_lengths)
            max_length = max(max_length, token_len)

        logging.info(f'Max word length: {max_length}')
        return max_length + 4
