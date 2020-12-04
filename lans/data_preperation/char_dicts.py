import logging
from typing import Dict

from lans.config import JOINT_LABELS_COL
from lans.constants import TOKEN_STR, EOW, SOW, NULL_LABEL
from tqdm import tqdm


def create_char2label_dict(df):
    dict, count = {}, 0
    for i, token_str in tqdm(enumerate(df[TOKEN_STR]), desc='Creating char2label dict', unit='token'):
        for char in token_str:
            if char not in dict:
                dict[char] = count
                count += 1

    dict[' '] = count
    dict[EOW] = count + 1
    dict[SOW] = count + 2
    logging.info(f"Number of charecters in language: {len(dict)}")
    return dict

def create_pos_encoder(df) -> Dict:
    pos_dict, count = {}, 1
    pos_dict[NULL_LABEL] = 0
    for pos in tqdm(df[JOINT_LABELS_COL], desc='Creating char2label dict', unit='token'):
        if pos not in pos_dict:
            pos_dict[pos] = count
            count += 1

    return pos_dict