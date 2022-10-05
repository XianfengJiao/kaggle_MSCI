from argparse import ArgumentParser
import pickle as pkl

def main(args):
    print('load multi_submission')
    multi_submission = pkl.load(open(args.multi_submission_path, 'rb'))
    print('load cite_submission')
    cite_submission = pkl.load(open(args.cite_submission_path, 'rb'))
    
    print('merge submission')
    multi_submission.reset_index(drop = True, inplace = True)
    multi_submission.index.name = 'row_id'
    multi_submission.iloc[:len(cite_submission.ravel())] = cite_submission.ravel()
    assert not multi_submission.isna().any()
    print('save submission')
    multi_submission.to_csv(args.save_path)
    print('end saving submission')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cite_submission_path', default='/home/jxf/code/kaggle_MSCI/results/20220928-13_mlp_i120+144_h1024_b512_lr1e-3_withExpert_zscore_cite.pkl', type=str)
    parser.add_argument('--multi_submission_path', default='/home/jxf/code/kaggle_MSCI/results/test_20220929-00_svd_mlp_i50_h512_b1024_lr1e-3_zscore_multi_multi_partial_submission_multiome.pkl', type=str)
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/results/0930_submission.csv', type=str)
    args = parser.parse_args()

    main(args)