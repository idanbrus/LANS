import sys

from lans.data_preperation.datasets.zeros_dataset import ZerosDataset

sys.path.append('../../../../')
from torch.utils.data import DataLoader

from lans.config import BATCH_SIZE
from lans.models.segmentors.base_segmentor import BaseSegmentor


class ZerosSegmentor(BaseSegmentor):

    def __init__(self, **kwargs):
        super(ZerosSegmentor, self).__init__(embedding_dim=1024, **kwargs)

    def _create_embedder(self):
        return

    def _create_dataloader(self, df):
        dataset = ZerosDataset(df, char2label_dict=self.char_dict, token_embedder=self._create_embedder(),
                               max_word_length=self.max_word_length)
        return DataLoader(dataset, batch_size=BATCH_SIZE)
