from typing import Dict

import pandas as pd
import torch

from lans.config import GPU_ID, BERT_CHECKPOINT
from lans.data_preperation.datasets.joint.base_joint_dataset import BaseJointDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoConfig

from lans.constants import TOKEN_STR, TOKEN_ID, SENT_ID


class BertJointDataset(BaseJointDataset):
    def __init__(self, df: pd.DataFrame, char2label_dict: Dict[str, int], pos_encoder: Dict[str, int], token_embedder,
                 max_word_length: int, model_checkpoint: str = BERT_CHECKPOINT):

        config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.embedder = AutoModel.from_pretrained(model_checkpoint, config=config).to(GPU_ID)
        super().__init__(df, char2label_dict, pos_encoder, token_embedder, max_word_length)

    def _calculate_token_embeddings(self, df: pd.DataFrame, embedder: BertModel):
        all_embeddings = []
        for id, sentence_df in tqdm(df.groupby(SENT_ID), desc='Creating Bert Embeddings', unit='sentence'):
            tokens_list = list(sentence_df.groupby(TOKEN_ID).first()[TOKEN_STR])
            sentence = ' '.join(tokens_list)

            input_ids = self.tokenizer.encode(sentence)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(GPU_ID)
            token_embeddings = self._creat_embeddings(embedder, input_ids).to('cpu')

            sent_emb = self._untokenize(tokens_list, token_embeddings)
            all_embeddings.append(sent_emb.data.numpy())

        return all_embeddings

    def _creat_embeddings(self, embedder, input_ids):
        max_tokens = 512
        if input_ids.shape[1] > max_tokens:
            part1 = embedder(input_ids[:, :max_tokens])[0].squeeze()
            part2 = embedder(input_ids[:, max_tokens:])[0].squeeze()
            return torch.cat([part1, part2])
        else:
            return embedder(input_ids)[0].squeeze()

    def _untokenize(self, words_list, token_embeddings):
        embeddings, i = [], 1

        for word in words_list:
            n_tokens = len(self.tokenizer.tokenize(word))
            word_embddings = token_embeddings[i:i + n_tokens].mean(axis=0)
            embeddings.append(word_embddings)
            i += n_tokens

        embeddings = torch.stack(embeddings)
        return embeddings

    def _create_token_embedding(self, sent_id, token_id):
        first_sent_id = self.group_tokens[0][0][0]
        sentence_index = sent_id - first_sent_id
        token_embedding = torch.from_numpy(self.token_embeddings[sentence_index][token_id - 1])
        return token_embedding
