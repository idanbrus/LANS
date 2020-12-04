from lans.models.segmentors.bert_segmentor import BertSegmentor
from lans.models.segmentors.fasttext_segmentor import FasttextSegmentor
from lans.models.segmentors.joint.bert_joint_segmentor import BertJointSegmentor
from lans.models.segmentors.zeros_segmentor import ZerosSegmentor

CONTEXT_OPTIONS = {'zeros': ZerosSegmentor,
                   'bert': BertSegmentor,
                   'fasttext': FasttextSegmentor,
                   'bert_joint': BertJointSegmentor}
