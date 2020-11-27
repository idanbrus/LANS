from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig

from lans.config import BATCH_SIZE, BERT_CHECKPOINT
from lans.data_preperation.datasets.bert_dataset import BertDataset
from lans.models.segmentors.base_segmentor import BaseSegmentor


class BertSegmentor(BaseSegmentor):

    def __init__(self, **kwargs):
        self.model_checkpoint = BERT_CHECKPOINT

        super(BertSegmentor, self).__init__(embedding_dim=768, **kwargs)

    def _create_embedder(self):
        config = AutoConfig.from_pretrained(self.model_checkpoint, output_hidden_states=True)
        embedder = AutoModel.from_pretrained(self.model_checkpoint, config=config)
        return embedder

    def _create_dataloader(self, df):
        dataset = BertDataset(df, model_checkpoint=self.model_checkpoint, char2label_dict=self.char_dict,
                              token_embedder=self._create_embedder(), max_word_length=self.max_word_length)
        return DataLoader(dataset, batch_size=BATCH_SIZE)
