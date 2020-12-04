from collections import Counter
from typing import List


def word_level_f_score(true_sentences: List[List[str]], pred_sentences: List[List[str]]) -> float:
    tp, true_count, pred_count = 0, 0, 0
    for true_sent, pred_sent in zip(true_sentences, pred_sentences):
        sent_tp, sent_true, sent_pred = score(true_sent, pred_sent)
        tp += sent_tp
        true_count += sent_true
        pred_count += sent_pred

    recall = tp / true_count if true_count > 0 else 0
    precision = tp / pred_count if pred_count > 0 else 0
    if recall + precision == 0:
        return 0
    f_score = 2 * precision * recall / (recall + precision)
    return f_score


# function Taken from the paper https://github.com/yanshao9798/segmenter/blob/master/evaluation.py
def score(gtokens, stokens):
    lcs = {}
    for i in range(0, len(gtokens) + 1):
        lcs[(i, 0)] = 0
    for j in range(0, len(stokens) + 1):
        lcs[(0, j)] = 0
    for i in range(1, len(gtokens) + 1):
        for j in range(1, len(stokens) + 1):
            if eq_tokens(gtokens[i - 1], stokens[j - 1]):
                lcs[(i, j)] = lcs[(i - 1, j - 1)] + 1
            else:
                if lcs[(i - 1, j)] >= lcs[(i, j - 1)]:
                    lcs[(i, j)] = lcs[(i - 1, j)]
                else:
                    lcs[(i, j)] = lcs[(i, j - 1)]

    tp = lcs[(len(gtokens), len(stokens))]
    g = len(gtokens)
    s = len(stokens)

    return tp, g, s

# function Taken from the paper https://github.com/yanshao9798/segmenter/blob/master/evaluation.py
def eq_tokens(gt, st):
    if gt.strip() in ["``", "''"] and st.strip() == '"':
        return True
    else:
        return gt == st

def pos_tag_eval(gold_df, pred_df):
    gold_gb = gold_df.groupby([gold_df.sent_id])
    pred_gb = pred_df.groupby([pred_df.sent_id])
    gold_counts, pred_counts, intersection_counts = 0, 0, 0
    for sent_id, gold in sorted(gold_gb):
        pred = pred_gb.get_group((sent_id))
        gold_seg_tags = list(zip(gold.upostag.tolist(), gold.form.tolist()))
        pred_seg_tags = list(zip(pred.upostag.tolist(), pred.form.tolist()))
        gold_count, pred_count = Counter(gold_seg_tags), Counter(pred_seg_tags)
        intersection_count = gold_count & pred_count
        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())
    precision = intersection_counts / pred_counts if pred_counts else 0.0
    recall = intersection_counts / gold_counts if gold_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1