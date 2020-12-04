import argparse
import os

import pandas as pd
import torch

from lans.config import GPU_ID, EMB_TYPE, PREPROCESSING_CACHE_DIR
from lans.data_preperation.tree_bank_loader import preprocess_conll_file
from lans.models.segmentors.context_dict import CONTEXT_OPTIONS
from lans.utils import txt2df, get_sent_lengths, tensor2sentence, save_pred_sentences


def main(test_path: str, train_path:str, results_dir: str, checkpoint: str):
    train_df = preprocess_conll_file(train_path,
                                                 cache_path=os.path.join(PREPROCESSING_CACHE_DIR, 'train.csv'))
    # df = pd.DataFrame()
    test_df = txt2df(test_path)
    model = CONTEXT_OPTIONS[EMB_TYPE].load_from_checkpoint(checkpoint, train_df=train_df, dev_df=pd.DataFrame(), test_df=pd.DataFrame(), exp_name='')

    test_dataloader = model._create_dataloader(test_df)
    device = torch.device(GPU_ID)
    model.to(device)

    # predict
    y_pred = []
    for token_embeddings, token_chars, y_true in test_dataloader:
        if EMB_TYPE == 'fasttext':
            token_embeddings = [emb.to(device) for emb in token_embeddings] # only for fasttext.
        else:
            token_embeddings = token_embeddings.to(device)  # not fasttext
        _, y_pred_batch = model((token_embeddings, token_chars.to(device), y_true.to(device)))
        y_pred.append(y_pred_batch.to('cpu'))
    y_pred = torch.cat(y_pred)

    # reconstruct results
    sent_lens = get_sent_lengths(test_df)
    pred_sentences = [tensor2sentence(sentence, model.char_dict) for sentence in torch.split(y_pred, sent_lens)]

    # convert sentences to df for the tagger
    save_pred_sentences(pred_sentences, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--test_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()

    main(args.test_path, args.train_path, args.results_dir, args.checkpoint)
