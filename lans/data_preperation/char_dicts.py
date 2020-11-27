import logging

from lans.constants import TOKEN_STR, EOW, SOW
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
