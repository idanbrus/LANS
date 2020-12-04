from typing import Dict

import pandas as pd
import torch
from lans.config import BERT_CHECKPOINT, GPU_ID
from tqdm import tqdm
from transformers import BertModel, AutoModel, AutoTokenizer, AutoConfig

from lans.constants import TOKEN_STR, TOKEN_ID, SENT_ID
from lans.data_preperation.datasets.base_dataset import BaseDataset


class BertDataset(BaseDataset):

    def __init__(self, df: pd.DataFrame, char2label_dict: Dict[str, int], token_embedder, max_word_length: int,
                 model_checkpoint: str = BERT_CHECKPOINT):
        config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.embedder = AutoModel.from_pretrained(model_checkpoint, config=config).to(GPU_ID)
        super().__init__(df, char2label_dict, token_embedder, max_word_length)

    def _calculate_token_embeddings(self, df: pd.DataFrame, embedder: BertModel):


        all_embeddings = []
        for id, sentence_df in tqdm(df.groupby(SENT_ID), desc='Creating Bert Embeddings', unit='sentence'):
            tokens_list = list(sentence_df.groupby(TOKEN_ID).first()[TOKEN_STR])
            sentence = ' '.join(tokens_list)

            input_ids = self.tokenizer.encode(sentence)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(GPU_ID) # .to('cpu')
            token_embeddings = self._creat_embeddings(input_ids).to('cpu')
            sent_emb = self._untokenize(tokens_list, token_embeddings)
            # sent_emb = self._untokenize_bpe(tokens_list, token_embeddings)
            all_embeddings.append(sent_emb.data.numpy())

        embedder.to('cpu')
        return all_embeddings

    def _get_token_embedding(self, sent_id, token_id):
        first_sent_id = self.group_tokens[0][0][0]
        sentence_index = sent_id - first_sent_id
        token_embedding = torch.from_numpy(self.token_embeddings[sentence_index][token_id - 1])
        return token_embedding

    def _creat_embeddings(self, input_ids):
        max_tokens = 512
        # input_ids
        if input_ids.shape[1] > max_tokens:
            part1 = self.embedder(input_ids[:, :max_tokens])[0].squeeze()
            part2 = self.embedder(input_ids[:, max_tokens:])[0].squeeze()
            return torch.cat([part1, part2])
        else:
            return self.embedder(input_ids)[0].squeeze()

    def _untokenize(self, words_list, token_embeddings):
        embeddings, i = [], 1

        for word in words_list:
            n_tokens = len(self.tokenizer.tokenize(word))
            word_embddings = token_embeddings[i:i + n_tokens].mean(axis=0)
            embeddings.append(word_embddings)
            i += n_tokens

        embeddings = torch.stack(embeddings)
        return embeddings

    def _untokenize_bpe(self, words_list, token_embeddings):
        embeddings, i = [], 0
        for word in words_list:
            word = ' ' + word if i != 0 else word
            n_tokens = len(self.tokenizer.tokenize(word))
            word_embddings = token_embeddings[i:i + n_tokens].mean(axis=0)
            embeddings.append(word_embddings)
            i += n_tokens

        embeddings = torch.stack(embeddings)
        return embeddings