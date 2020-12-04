import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from lans.config import BATCH_SIZE, BERT_CHECKPOINT, GPU_ID
from lans.data_preperation.datasets.joint.bert_joint_dataset import BertJointDataset
from lans.models.segmentors.joint.base_joint_segmentor import BaseJointSegmentor


class BertJointSegmentor(BaseJointSegmentor):

    def __init__(self, **kwargs):
        self.model_checkpoint = BERT_CHECKPOINT

        super(BertJointSegmentor, self).__init__(embedding_dim=768, **kwargs)

    def _create_embedder(self):
        config = AutoConfig.from_pretrained(self.model_checkpoint, output_hidden_states=True)
        embedder = AutoModel.from_pretrained(self.model_checkpoint, config=config).to(GPU_ID)
        return embedder

    def _create_dataloader(self, df):
        dataset = BertJointDataset(df, char2label_dict=self.char_dict, pos_encoder=self.pos_dict,
                                   token_embedder=self._create_embedder(),
                                   max_word_length=self.max_word_length)
        return DataLoader(dataset, batch_size=BATCH_SIZE)
