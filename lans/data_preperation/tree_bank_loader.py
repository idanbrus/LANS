import logging
import os
from tqdm import tqdm
import pandas as pd
import re

from lans.config import PREPROCESSING_CACHE_DIR


def preprocess_conll_file(path:str, cache_path:str = None):

    if PREPROCESSING_CACHE_DIR is not None:
        if os.path.isfile(cache_path):
            logging.info(f'Reading cached preprocessed: {cache_path}')
            return pd.read_csv(cache_path)

    new_df = _preprocess_conll_file(path)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        new_df.to_csv(cache_path)
        logging.info(f'Saving cached preprocessed: {cache_path}')
    return new_df


def _preprocess_conll_file(path):
    df = _read_conll(path)
    df = df.fillna('.')
    df['form'] = df['form'].apply(lambda form: str(form).replace('_', '') if form != '_' else form).dropna()
    first_sent_id = int(_get_first_sent_id(path))
    df = _add_fields_to_df(df, first_sent_id)

    return df

# Function Code by https://github.com/cjer/bclm
def _read_conll(path, add_head_stuff=False):
    # CoNLL file is tab delimeted with no quoting
    # quoting=3 is csv.QUOTE_NONE
    df = (pd.read_csv(path, sep='\t', header=None, quoting=3, comment='#', encoding='utf-8',
                names = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'])
                # add sentence labels
                .assign(sent_id = lambda x: (x.id==1).cumsum())
                # replace bad root dependency tags
                .replace({'deprel': {'prd': 'ROOT'}})
               )
    if add_head_stuff:
        df = df.merge(df[['id', 'form', 'sent', 'upostag']].rename(index=str, columns={'form': 'head_form', 'upostag': 'head_upos'}).set_index(['sent', 'id']),
               left_on=['sent', 'head'], right_index=True, how='left')
    return df

def _add_fields_to_df(df, first_sent_id):

    sent_id = first_sent_id
    token_id = 1
    token_str = ""
    words_in_token = 0

    df['token_id'], df['token_str'] = None, None

    for i in tqdm(range(len(df)), desc='preprocessing conllu file', unit='token'):
    # for i in range(len(df)):
        # add sentence id
        if str(df.loc[i, 'id']) in ['1','1.0']:
            sent_id += 1
            token_id = 1
        df.loc[i, 'sent_id'] = sent_id

        # add token str
        try:
            int(df.loc[i, 'id'])
            if words_in_token == 0:  # not a multiword
                df.loc[i, 'token_str'] = df.loc[i, 'form']
                df.loc[i, 'token_id'] = token_id
                token_id += 1
            else:
                df.loc[i, 'token_str'] = token_str
                df.loc[i, 'token_id'] = token_id
                words_in_token -= 1
                token_id = token_id + 1 if words_in_token == 0 else token_id
        except:  # beginning of a multi word
            start_word, end_word = re.findall('[0-9]+', df.loc[i, 'id'])
            token_str = df.loc[i, 'form']
            words_in_token = int(end_word) - int(start_word) + 1
            df = df.drop(i, axis=0)
    return df

def _get_first_sent_id(path):
    with open(path, 'r', encoding="utf8") as f:
        first_line = f.readline()
        first_id = re.findall('[0-9]+$', first_line)[0]
        return first_id