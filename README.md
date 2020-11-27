# LANS
Language Agnostic Neural Segmentation base on paper: <ADD_URL>

## Installation
use `conda env create -f environment.yml`

## Training a new model
* example: `python train.py --train_path .\lans\example_data\ud_Hebrew-HTB\he_htb-ud-train.conllu 
--dev_path .\lans\example_data\ud_Hebrew-HTB\he_htb-ud-dev.conllu --test_path .\lans\example_data\ud_Hebrew-HTB\he_htb-ud-test.conllu`
* Currenly all 3 files (train, dev and test) of type `.conll` must be provided.
* Change training configuration in `lans\config.py` including:
    * context configuration (currently only `zeros` and `bert` supported)
    * BERT checkpoint (relevant if `bert` context was chosen)
    * GPU number
    * Experiment name (can be overridden on command line `--experiment_name`)
* Tensorboard file is generated.
* Predictions will be saved as a `.txt` file

##### Warnings
* Not changing experiment name between runs might overwrite both the saved model and the predictions.
* There is a long pre-processing stage, which is cached in `PREPROCESSING_CACHE_DIR`.
 Whenever changing the dataset (train, dev or test) make sure you change `PREPROCESSING_CACHE_DIR`
  (Change it to `None` if no cache is desired).
 
## Predicting using model checkpoint
* example: `--checkpoint .\lans\saved_checkpoints\example.ckpt
--test_path
.\lans\example_data\ud_test_tokens.txt
--train_path C:\.\lans\example_data\UD_Hebrew-HTB\he_htb-ud-train.conllu
--results_dir .\predictions\example_prediction`


#### Warnings
* The train path is currently still needed to create the character dictionary
* Same `lans\config.py` must be used as in training. (e.g don't change hidden size)


