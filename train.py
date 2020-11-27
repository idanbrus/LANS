import argparse
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from lans.config import MAX_EPOCHS, GPU_ID, TENSORBOARD_DIR, EMB_TYPE, SAVED_MODEL_PATH, \
    PREPROCESSING_CACHE_DIR, CHECK_VAL_EVERY, EXPERIMENT_NAME
from lans.data_preperation.tree_bank_loader import preprocess_conll_file
from lans.models.segmentors.bert_segmentor import BertSegmentor
from lans.models.segmentors.zeros_segmentor import ZerosSegmentor

segmentor_options = {'zeros': ZerosSegmentor,
                     'bert': BertSegmentor}


def main(train_path, dev_path, test_path, experiment_name):
    post_processed_train = preprocess_conll_file(train_path,
                                                 cache_path=os.path.join(PREPROCESSING_CACHE_DIR, 'train.csv'))
    post_processed_dev = preprocess_conll_file(dev_path,
                                               cache_path=os.path.join(PREPROCESSING_CACHE_DIR,
                                                                       'dev.csv'))
    post_processed_test = preprocess_conll_file(test_path,
                                                cache_path=os.path.join(PREPROCESSING_CACHE_DIR,
                                                                        'test.csv'))

    logging.info(f'Running Experiment: {experiment_name}')
    segmentor = segmentor_options[EMB_TYPE](train_df=post_processed_train,
                                            dev_df=post_processed_dev,
                                            test_df=post_processed_test,
                                            exp_name=experiment_name)

    logger = TensorBoardLogger(TENSORBOARD_DIR, name=experiment_name)
    trainer = Trainer(gpus=str(GPU_ID), logger=logger, max_epochs=MAX_EPOCHS, check_val_every_n_epoch=CHECK_VAL_EVERY)

    trainer.fit(segmentor)
    trainer.save_checkpoint(os.path.join(SAVED_MODEL_PATH, f'{experiment_name}.ckpt'))
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--train_path', type=str, help='path to train conllu file')
    parser.add_argument('--dev_path', type=str, help='path to dev conllu file')
    parser.add_argument('--test_path', type=str, help='path to test conllu file')
    parser.add_argument('--experiment_name', type=str, help='path to test conllu file', default=EXPERIMENT_NAME)

    args = parser.parse_args()

    if not args.train_path:
        raise Exception('Must Include train path')
    main(args.train_path, args.dev_path, args.test_path, args.experiment_name)
