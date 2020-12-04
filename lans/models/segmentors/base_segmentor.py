import abc

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam

from lans.config import LEARNING_RATE, DROPOUT_P, GPU_ID, HIDDEN_SIZE, MAX_INPUT_SIZE
from lans.constants import SOW, EOW
from lans.data_preperation.char_dicts import create_char2label_dict
from lans.data_preperation.datasets.base_dataset import BaseDataset
from lans.evaluation.metrics import word_level_f_score
from lans.models.attention_decoder import AttnDecoderRNN
from lans.models.encoder import Encoder
from lans.utils import tensor2sentence, get_sent_lengths, save_pred_sentences


class BaseSegmentor(LightningModule, abc.ABC):
    def __init__(self, train_df: pd.DataFrame,
                 dev_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 exp_name: str,
                 hidden_size: int = HIDDEN_SIZE,
                 input_size: int = MAX_INPUT_SIZE,
                 dropout_p: float = DROPOUT_P,
                 gpu_num: int = GPU_ID,
                 embedding_dim: int = 512):
        super().__init__()

        self.train_df = train_df
        self.val_df = dev_df
        self.test_df = test_df
        self.exp_name = exp_name
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.gpu_id = gpu_num
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        # prepare the data
        self.char_dict = create_char2label_dict(self.train_df)
        self.max_word_length = self.find_max_word_length(self.train_df)

        # encoder-decoder model
        self.encoder = Encoder(len(self.char_dict), input_size, self.hidden_size, embedding_dim=embedding_dim)
        self.attn_decoder = AttnDecoderRNN(hidden_size, len(self.char_dict), dropout_p=dropout_p,
                                           max_length=self.max_word_length)

        self.loss = torch.nn.NLLLoss()

    @abc.abstractmethod
    def _create_embedder(self):
        raise NotImplementedError

    def find_max_word_length(self, df):
        return BaseDataset.find_max_word_length(df)

    def prepare_data(self):
        self.train_sentence_lens = get_sent_lengths(self.train_df)
        self.val_sentence_lens = get_sent_lengths(self.val_df)
        self.test_sentence_lens = get_sent_lengths(self.test_df)

    def training_step(self, batch, batch_idx):
        loss, y_pred = self.forward(batch, train=True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.forward(batch)
        _, _, y_true = batch
        return {'output': (y_true, y_pred), 'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, y_pred = self.forward(batch)
        _, _, y_true = batch
        return {'output': (y_true, y_pred), 'test_loss': loss}

    def forward(self, batch, train=False):
        token_embeddings, token_chars, y = batch

        # encoder
        encoder_output, (hidden, cell_state) = self.encoder(token_chars, token_embeddings)

        # decoder
        cell_state, pred_chars = torch.zeros_like(hidden), torch.ones_like(y[:, 0]) * self.char_dict[SOW]
        max_seq_len = y.shape[1]
        loss, y_pred = 0, []
        for i in range(max_seq_len):
            if (pred_chars != self.char_dict[EOW]).any():  # reached the end of all the words
                output, hidden, cell_state, attn_weights = self.attn_decoder(pred_chars, hidden, cell_state,
                                                                             encoder_output)
                loss += self.loss(output, y[:, i])
            pred_chars = torch.argmax(output, dim=-1)  # predicted character
            y_pred.append(pred_chars)
            if train:  # use true y while training
                pred_chars = y[:, i]
        loss = loss / max_seq_len
        y_pred = torch.stack(y_pred, dim=1)

        return loss, y_pred

    def validation_epoch_end(self, outputs, is_test: bool = False):
        y_true, y_pred = [], []
        model_outputs = [output['output'] for output in outputs]
        for batch_y_true, batch_y_pred in model_outputs:
            y_true.append(batch_y_true)
            y_pred.append(batch_y_pred)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        loss_key = 'test_loss' if is_test else 'val_loss'
        avg_loss = torch.stack([x[loss_key] for x in outputs]).mean()

        accuracy, f_score = self._evaluation_metrics(y_true, y_pred, test=is_test)
        return {'log': {r'letter_accuracy': accuracy, r'f_score': f_score, r'avg_val_loss': avg_loss},
                'val_loss': avg_loss,
                'y_true_pred': {'y_pred': y_pred, 'y_true': y_true}}

    def test_epoch_end(self, outputs):
        results = self.validation_epoch_end(outputs, is_test=True)
        print(f"Test Results: {results['log']['f_score']}\t char_accuracy:{results['log']['letter_accuracy']}")
        return results

    def _evaluation_metrics(self, y_true, y_pred, test=False):
        # charecter level accuracy
        char_mask = y_true != self.char_dict[EOW]
        is_same = y_true == y_pred
        accuracy = (is_same * char_mask).sum().item() / char_mask.sum().item()

        # full sentence metrics
        # Just for Validation sanity check
        sentence_lens = self.val_sentence_lens if sum(self.val_sentence_lens) == len(y_true) else [len(y_true)]
        sentence_lens = self.test_sentence_lens if test else sentence_lens

        # comment out for better running time
        true_sentences = [tensor2sentence(sentence, self.char_dict) for sentence in torch.split(y_true, sentence_lens)]
        pred_sentences = [tensor2sentence(sentence, self.char_dict) for sentence in torch.split(y_pred, sentence_lens)]

        f_score = word_level_f_score(true_sentences, pred_sentences)

        save_pred_sentences(pred_sentences, self.exp_name)
        print(pred_sentences[0])
        return accuracy, f_score

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)

    def train_dataloader(self):
        return self._create_dataloader(self.train_df)

    def val_dataloader(self):
        return self._create_dataloader(self.val_df)

    def test_dataloader(self):
        return self._create_dataloader(self.test_df)

    @abc.abstractmethod
    def _create_dataloader(self, df:pd.DataFrame):
        raise NotImplementedError
