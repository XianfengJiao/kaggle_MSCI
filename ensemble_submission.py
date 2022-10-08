from argparse import ArgumentParser
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm

SUBMISSIONS = {
    '/home/jxf/code/kaggle_MSCI/results/20221007-23_mlp_i60+144_h512_b512_lr1e-3_withExpert_zscore_cite/kfold-0/submission_citeseq.pkl': 0.8,
    '/home/jxf/code/kaggle_MSCI/results/20221007-23_mlp_i60+144_h512_b512_lr1e-3_withExpert_zscore_cite/kfold-1/submission_citeseq.pkl': 0.9,
    '/home/jxf/code/kaggle_MSCI/results/20221007-23_mlp_i60+144_h512_b512_lr1e-3_withExpert_zscore_cite/kfold-2/submission_citeseq.pkl': 1.,
    '/home/jxf/code/kaggle_MSCI/results/20221007-23_mlp_i60+144_h512_b512_lr1e-3_withExpert_zscore_cite/kfold-3/submission_citeseq.pkl': 1.,
    '/home/jxf/code/kaggle_MSCI/results/20221007-23_mlp_i60+144_h512_b512_lr1e-3_withExpert_zscore_cite/kfold-4/submission_citeseq.pkl': 1.,
    '/home/jxf/code/kaggle_MSCI/results/20220929-02_mlp_i120+144_h1024_b512_lr5e-3_withExpert_zscore_cite.pkl': 0.8,
    '/home/jxf/code/kaggle_MSCI/results/20220906-23_test_mlp_i50_h512_b512_lr1e3_zscore_multi_partial_submission_multiome.pkl': 0.8,
    '/home/jxf/code/kaggle_MSCI/results/20220906-17_mlp_i4096_h8192_b512_lr1e4_zscore_partial_submission_multiome.pkl': 0.9,
    '/home/jxf/code/kaggle_MSCI/results/20220906-17_mlp_i50_h512_b512_lr1e3_zscore_partial_submission_multiome.pkl': 0.8,
    '/home/jxf/code/kaggle_MSCI/results/test_20220929-00_svd_mlp_i50_h512_b1024_lr1e-3_zscore_multi_multi_partial_submission_multiome.pkl': 1,
    }

cell_ids = pd.read_parquet('/home/jxf/code/kaggle_MSCI/input/evaluation.parquet').cell_id
PRED_SEGMENTS = [(0, 6812820), (6812820, 65744180)]


def get_std_cite(submission_path):
    submission = pkl.load(open(submission_path, 'rb'))
    return (submission - np.expand_dims(np.mean(submission, axis=1), axis=-1)) / np.expand_dims(np.std(submission, axis=1), axis=-1)

def get_std_multi(submission_path):
    """
    Standardize submission per cell_id
    """
    df = pd.read_pickle(submission_path)
    df.reset_index(drop=True, inplace=True)
    df['cell_id'] = cell_ids
    vals = []
    for idx, g in tqdm(df.groupby('cell_id', sort=False), desc=f'Standardizing {submission_path}', miniters=1000):
        val = (g.values - np.mean(g.values)) / np.std(g.values)
        vals.append(val.flatten())
    vals = np.concatenate(vals)
    return vals

def gen_ensemble(technology, gen_submission):
    ensemble = None
    for path in tqdm([path for path in SUBMISSIONS.keys() if technology in path], desc='Process submission'):
        weight = SUBMISSIONS[path]
        if ensemble is None:
            ensemble = gen_submission(path) * weight
        else:
            ensemble += gen_submission(path) * weight
    return ensemble


def main(args):
    ensemble = []
    for tech, (from_idx, to_idx), std_func in tqdm(list(zip(['cite', 'multiome'], PRED_SEGMENTS, [get_std_cite, get_std_multi])), desc='Technology'):
        ensemble.append(gen_ensemble(tech, std_func).ravel()[from_idx: to_idx])
    ensemble = np.concatenate(ensemble)
    df_submit = pd.read_parquet('/home/jxf/code/kaggle_MSCI/input/sample_submission.parquet')
    df_submit['target'] = ensemble
    df_submit.to_csv(args.save_path, index=False)
    df_submit
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/results/1008_submission.csv', type=str)
    args = parser.parse_args()
    main(args)