import os
from shutil import copyfile
from typing import Dict, List

import torch

from lans.config import PREDICTION_OUTPUT_DIR, ROOT_DIR, JOINT_LABELS_COL
from lans.constants import EOW, TOKEN_ID, SENT_ID, FORM, NULL_LABEL
import pandas as pd
import csv
from tqdm import tqdm

def tensor2sentence(tensor: torch.Tensor, char_dict: Dict):
    key_dict = {value: key for key, value in char_dict.items()}
    all_tokens = []
    eow_key = char_dict[EOW]
    for token in tensor:
        token = ''.join([key_dict[char.item()] for char in token if char != eow_key and char.item() in key_dict.keys()])
        token = token[:-1] if token.endswith(' ') else token
        all_tokens += token.split(' ')
    return all_tokens

def get_sent_lengths(df) -> List[int]:
    lengths = [grouped_df[TOKEN_ID].max() for sent_id, grouped_df in df.groupby(SENT_ID)]
    return lengths

def save_pred_sentences(pred_sentences: List[List[str]], exp_name:str):
    path = os.path.join(PREDICTION_OUTPUT_DIR, exp_name, 'predictions.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w+').close()  # clears the file
    with open(path, 'a+', encoding='utf-8') as f:
        for sentence in pred_sentences:
            for word in sentence:
                f.write(word + '\n')
            f.write('\n')

    ## save config file
    config_path = os.path.join(ROOT_DIR, 'lans', 'config.py')
    config_save_path = os.path.join(PREDICTION_OUTPUT_DIR, exp_name, 'config.py')
    copyfile(config_path, config_save_path)


def txt2df(path) -> pd.DataFrame:
    df = pd.read_csv(path, names=['form'], sep=' ', header=None, skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    df[SENT_ID] = pd.NaT
    df['token_id'] = pd.NaT
    sent_id, token_id = 0, 1
    for i in tqdm(range(len(df)), desc='Converting txt to pandas Dataframe', unit='token'):
        df.loc[i, SENT_ID] = sent_id
        df.loc[i, 'token_id'] = token_id
        token_id += 1
        if df.loc[i].isna().any():
            df.drop(i, inplace=True)
            sent_id += 1
            token_id = 1
    df['token_str'] = df[FORM]
    return df

def creat_pos_df(tokens_list: List[List[List[str]]], pos_list: List[List[List[str]]]) -> pd.DataFrame:
    df = pd.DataFrame(columns=[FORM, JOINT_LABELS_COL, SENT_ID])
    for sent_id, (tokens, pos_tags) in enumerate(zip(tokens_list, pos_list)):
        tmp_df = _create_token_pos_df(tokens, pos_tags, sent_id)
        df = df.append(tmp_df)
    return df

def _create_token_pos_df(tokens_list: List[List[str]], pos_list: List[List[str]], sent_id) -> pd.DataFrame:
    df = pd.DataFrame(columns=[FORM, JOINT_LABELS_COL, SENT_ID])
    for tokens, pos_tags in zip(tokens_list, pos_list):
        if len(pos_tags) > len(tokens):
            pos_tags = pos_tags[:len(tokens)]

        elif len(pos_tags) < len(tokens):
            pos_tags = pos_tags + [NULL_LABEL] * (len(tokens) - len(pos_tags))
        tmp_df = pd.DataFrame({FORM: tokens, JOINT_LABELS_COL: pos_tags, SENT_ID: sent_id})
        df = df.append(tmp_df)
    return df

def tensor2token_pos(pos_tensor: torch.Tensor, tokens_tensor: torch.Tensor, space_id: int, pos_dict: Dict):
    key_dict = {value: key for key, value in pos_dict.items()}
    mask = tokens_tensor == space_id
    all_pos_tags = []
    for token_pos, token_mask in zip(pos_tensor, mask):
        pos = [key_dict[pos_id.item()] for pos_id, cur_mask in zip(token_pos, token_mask) if cur_mask]
        all_pos_tags.append(pos)
    return all_pos_tags