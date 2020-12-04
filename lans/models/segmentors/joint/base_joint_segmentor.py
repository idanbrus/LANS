from functools import reduce

import pandas as pd
import torch

from lans.config import POS_WEIGHT
from lans.constants import EOW, SOW
from lans.data_preperation.char_dicts import create_pos_encoder
from lans.evaluation.metrics import word_level_f_score, pos_tag_eval
from lans.models.segmentors.base_segmentor import BaseSegmentor
from lans.utils import tensor2token_pos, creat_pos_df


class BaseJointSegmentor(BaseSegmentor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_dict = create_pos_encoder(self.train_df)

        self.pos_classifier = torch.nn.Sequential(torch.nn.Linear(kwargs['embedding_dim'] + self.hidden_size, len(self.pos_dict)),
                                                  torch.nn.LogSoftmax(dim=1))
        self.pos_loss = torch.nn.NLLLoss(reduction='none')

    def forward(self, batch, train=False):
        token_embeddings, token_chars, y, pos_tags = batch

        # encoder
        encoder_output, (hidden, cell_state) = self.encoder(token_chars, token_embeddings)

        # decoder
        cell_state, pred_chars = torch.zeros_like(hidden), torch.ones_like(y[:, 0]) * self.char_dict[SOW]
        max_seq_len = y.shape[1]
        loss, pos_loss, y_pred, pos_pred = 0, 0, [], []
        for i in range(max_seq_len):
            if (pred_chars != self.char_dict[EOW]).any():  # reached the end of all the words
                output, hidden, cell_state, attn_weights = self.attn_decoder(pred_chars, hidden, cell_state,
                                                                             encoder_output)
                loss += self.loss(output, y[:, i])

            pred_chars = torch.argmax(output, dim=-1)  # predicted character
            y_pred.append(pred_chars)
            if train:  # use true y while training
                pred_chars = y[:, i]

            # POS
            mask = pred_chars == self.char_dict[' ']

            pos_input = torch.cat((hidden.squeeze(), token_embeddings), dim=1)
            pos_probas = self.pos_classifier(pos_input).squeeze()
            pos_loss += (self.pos_loss(pos_probas, pos_tags[:, i]) * mask).mean()
            batch_pos_pred = torch.argmax(pos_probas, dim=-1)
            batch_pos_pred = batch_pos_pred * mask
            pos_pred.append(batch_pos_pred)

        loss = (1 - POS_WEIGHT) * loss + POS_WEIGHT * pos_loss
        loss = loss / max_seq_len
        y_pred = torch.stack(y_pred, dim=1)
        pos_pred = torch.stack(pos_pred, dim=1)

        return loss, y_pred, pos_pred

    def training_step(self, batch, batch_idx):
        loss, y_pred, pos_pred = self.forward(batch, train=True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss, y_pred, pos_pred = self.forward(batch)
        _, _, y_true, pos_true = batch
        return {'output': (y_true, y_pred, pos_true, pos_pred), 'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, y_pred, pos_pred = self.forward(batch)
        _, _, y_true, pos_true = batch
        return {'output': (y_true, y_pred, pos_true, pos_pred), 'test_loss': loss}

    def validation_epoch_end(self, outputs, is_test: bool = False):
        y_true, y_pred, pos_true, pos_pred = [], [], [], []
        model_outputs = [output['output'] for output in outputs]
        for batch_y_true, batch_y_pred, batch_pos_true, batch_pos_pred in model_outputs:
            y_true.append(batch_y_true)
            y_pred.append(batch_y_pred)
            pos_true.append(batch_pos_true)
            pos_pred.append(batch_pos_pred)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        pos_true = torch.cat(pos_true)
        pos_pred = torch.cat(pos_pred)

        loss_key = 'test_loss' if is_test else 'val_loss'
        avg_loss = torch.stack([x[loss_key] for x in outputs]).mean()

        accuracy, f_score, pos_f_score = self._evaluation_metrics(y_true, y_pred, pos_true, pos_pred, test=is_test)
        return {'log': {r'letter_accuracy': accuracy, r'f_score': f_score, r'avg_val_loss': avg_loss,
                        'POS_F_Score': pos_f_score},
                'val_loss': avg_loss,
                'y_true_pred': {'y_pred': y_pred, 'y_true': y_true}}

    def _evaluation_metrics(self, y_true, y_pred, pos_true, pos_pred, test=False):
        # charecter level accuracy
        char_mask = y_true != self.char_dict[EOW]
        is_same = y_true == y_pred
        accuracy = (is_same * char_mask).sum().item() / char_mask.sum().item()

        # full sentence metrics
        # Just for Validation sanity check
        sentence_lens = self.val_sentence_lens if sum(self.val_sentence_lens) == len(y_true) else [len(y_true)]
        sentence_lens = self.test_sentence_lens if test else sentence_lens

        ## New stuff
        true_sentences_tokens = [self._tensor2tokens(sentence, self.char_dict) for sentence in
                                 torch.split(y_true, sentence_lens)]
        pred_sentences_tokens = [self._tensor2tokens(sentence, self.char_dict) for sentence in
                                 torch.split(y_pred, sentence_lens)]

        true_sentences = [reduce(lambda x, y: x + y, sentence) for sentence in true_sentences_tokens]
        pred_sentences = [reduce(lambda x, y: x + y, sentence) for sentence in pred_sentences_tokens]
        f_score = word_level_f_score(true_sentences, pred_sentences)
        print(pred_sentences[0])

        space_id = self.char_dict[' ']
        true_pos = [tensor2token_pos(sent_pos, y, space_id, self.pos_dict) for sent_pos, y in
                    zip(torch.split(pos_true, sentence_lens), torch.split(y_true, sentence_lens))]
        pred_pos = [tensor2token_pos(sent_pos, y, space_id, self.pos_dict) for sent_pos, y in
                    zip(torch.split(pos_pred, sentence_lens), torch.split(y_pred, sentence_lens))]

        print(pred_pos[0])

        true_pos_df = creat_pos_df(true_sentences_tokens, true_pos)
        pred_pos_df = creat_pos_df(pred_sentences_tokens, pred_pos)
        # End new stuff

        pos_f_score = pos_tag_eval(true_pos_df, pred_pos_df)[2]

        return accuracy, f_score, pos_f_score

    def _tensor2tokens(self, tensor, char_dict):
        key_dict = {value: key for key, value in char_dict.items()}
        all_tokens = []
        eow_key = char_dict[EOW]
        for token in tensor:
            token = ''.join(
                [key_dict[char.item()] for char in token if char != eow_key and char.item() in key_dict.keys()])
            token = token[:-1] if token.endswith(' ') else token
            all_tokens.append(token.split(' '))
        return all_tokens
