import pandas as pd
import torch
from torch.utils.data import DataLoader

from lans.config import LANGUAGE
from lans.data_preperation.datasets.fasttext_dataset import FasttextDataset
from lans.models.encoder import Encoder
from lans.models.segmentors.base_segmentor import BaseSegmentor
import fasttext
import fasttext.util


class FasttextSegmentor(BaseSegmentor):
    language2id = {'hebrew': 'he',
                   'turkish': 'tr',
                   'arabic': 'ar',
                   'french': 'fr',
                   'english': 'en',
                   'german': 'de',
                   'russian': 'ru',
                   'kazakh': 'kk',
                   'spanish': 'es',
                   'latvian': 'lv',
                   'finnish': 'fi',
                   'czech': 'cs'}

    def __init__(self, **kwargs):
        super().__init__(embedding_dim =300, **kwargs)
        self.bilstm = torch.nn.LSTM(self.embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.encoder = Encoder(len(self.char_dict), self.input_size, self.hidden_size, embedding_dim=512)


    def _create_embedder(self):
        assert LANGUAGE in self.language2id, "Must set LANGUAGE in config.py file. if language doesn't exist in dict add it"
        language_id = self.language2id[LANGUAGE]
        fasttext.util.download_model(language_id, if_exists='ignore')
        return fasttext.load_model(f'cc.{language_id}.300.bin')

    def forward(self, batch, train=False):
        fastext, char_inputs, y = batch
        token_embs = [self.bilstm(sent)[0] for sent in fastext]
        token_embs = torch.cat(token_embs, dim=1).squeeze()

        return super().forward((token_embs, char_inputs, y), train=True)

    def _create_dataloader(self, df, batch_size=16):
        dataset = FasttextDataset(df, char2label_dict=self.char_dict, token_embedder=self._create_embedder(),
                                  max_word_length=self.max_word_length)

        def my_collate(batch):
            fasttext = [item[0].unsqueeze(0) for item in batch]

            x = torch.cat([item[1] for item in batch])
            y = torch.cat([item[2] for item in batch])
            return fasttext, x, y

        return DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate)

    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.forward(batch)
        _, _, y_true = batch
        return {'output': (y_true, y_pred), 'val_loss': loss}
