import torch

from lans.data_preperation.datasets.base_dataset import BaseDataset


class ZerosDataset(BaseDataset):

    def _calculate_token_embeddings(self, df, embedder):
        return

    def _get_token_embedding(self, sent_id, token_id):
        return torch.zeros(1024)





