from nltk.translate import meteor_score
from rouge_score import rouge_scorer, scoring
import numpy as np
import sacrebleu
from itertools import zip_longest


ref_Path = '../data_new/test/testOriginalSummary_5.txt'
ID_Path = '../data_new/test/testID_5.txt'
hyp_Path = '../results_new/helen5/generated.txt'


def meteor_evaluation(ref_path, hyp_path, ID_path):
    meteor_scores = list()
    reffile = open(ref_path, 'r')
    refs = [line.rstrip('\n') for line in reffile.readlines()]
    hypfile = open(hyp_path, 'r')
    hyps = [line.rstrip('\n') for line in hypfile.readlines()]
    IDfile = open(ID_path, 'r')
    IDs = [line for line in IDfile.read().splitlines()]
    unique_IDs = np.unique(np.array(IDs))
    for i,  unique_ID in enumerate(unique_IDs):
        indices = [i for i in range(len(IDs)) if IDs[i] == unique_ID]
        multi_ref = [refs[i] for i in indices]
        hyp = hyps[indices[0]]
        meteor_scores.append(meteor_score.meteor_score(multi_ref, hyp))
    meteor_score_avg = np.mean(meteor_scores)
    return round(meteor_score_avg, 5)

def rouge_evaluation(ref_path, hyp_path, ID_path):
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    rouge_scores = {}
    reffile = open(ref_path, 'r')
    refs = [line.rstrip('\n') for line in reffile.readlines()]
    hypfile = open(hyp_path, 'r')
    hyps = [line.rstrip('\n') for line in hypfile.readlines()]
    IDfile = open(ID_path, 'r')
    IDs = [line for line in IDfile.read().splitlines()]
    unique_IDs = np.unique(np.array(IDs))
    for i, unique_ID in enumerate(unique_IDs):
        indices = [i for i in range(len(IDs)) if IDs[i] == unique_ID]
        multi_ref = [refs[i] for i in indices]
        hyp = hyps[indices[0]]

        # ROUGE multi-ref jackknifing
        if len(multi_ref) > 1:
            cur_scores = [rouge.score(ref, hyp) for ref in multi_ref]

            # get best score for all leave-one-out sets
            best_scores = []
            for leave in range(len(multi_ref)):
                cur_scores_leave_one = [
                    cur_scores[s] for s in range(len(multi_ref)) if s != leave
                ]
                best_scores.append(
                    {
                        rouge_type: max(
                            [s[rouge_type] for s in cur_scores_leave_one],
                            key=lambda s: s.fmeasure,
                        )
                        for rouge_type in rouge_types
                    }
                )

            # average the leave-one-out bests to produce the final score
            score = {
                rouge_type: scoring.Score(
                    np.mean([b[rouge_type].precision for b in best_scores]),
                    np.mean([b[rouge_type].recall for b in best_scores]),
                    np.mean([b[rouge_type].fmeasure for b in best_scores]),
                )
                for rouge_type in rouge_types
            }
        else:
            score = rouge.score(refs[0], hyp)

        # convert the named tuples to plain nested dicts
        score = {
            rouge_type: {
                "precision": score[rouge_type].precision,
                "recall": score[rouge_type].recall,
                "fmeasure": score[rouge_type].fmeasure,
            }
            for rouge_type in rouge_types
        }
        rouge_scores[i] = score

    rouge_scores_values = list(rouge_scores.values())
    l1_keys = list(rouge_scores_values[0].keys())
    l2_keys = rouge_scores_values[0][l1_keys[0]].keys()
    return {
        key1: {
            key2: round(
                np.mean([score[key1][key2] for score in rouge_scores_values]), 5
            )
            for key2 in l2_keys
        }
        for key1 in l1_keys
    }

def bleu_evaluation(ref_path, hyp_path, ID_path):
    ref_dict = {}
    hyp_lst = list()
    reffile = open(ref_path, 'r')
    refs = [line.rstrip('\n') for line in reffile.readlines()]
    hypfile = open(hyp_path, 'r')
    hyps = [line.rstrip('\n') for line in hypfile.readlines()]
    IDfile = open(ID_path, 'r')
    IDs = [line for line in IDfile.read().splitlines()]
    unique_IDs = np.unique(np.array(IDs))
    for i,  unique_ID in enumerate(unique_IDs):
        indices = [i for i in range(len(IDs)) if IDs[i] == unique_ID]
        multi_ref = [refs[i] for i in indices]
        hyp = hyps[indices[0]]
        hyp_lst.append(hyp)
        ref_dict[unique_ID] = multi_ref

    ref_streams = list(zip_longest(*list(ref_dict.values())))
    bleu_score = sacrebleu.corpus_bleu(hyp_lst, ref_streams, lowercase=True)
    return round(bleu_score.score, 5)

meteor = meteor_evaluation(ref_Path, hyp_Path, ID_Path)
print("METEOR: " + str(meteor))
rouge = rouge_evaluation(ref_Path, hyp_Path, ID_Path)
print("ROUGE: " + str(rouge))
bleu = bleu_evaluation(ref_Path, hyp_Path, ID_Path)
print("BLEU: " + str(bleu))