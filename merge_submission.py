from argparse import ArgumentParser
import pickle as pkl

def main(args):
    print('load multi_submission')
    multi_submission = pkl.load(open(args.multi_submission_path, 'rb'))
    print('load cite_submission')
    cite_submission = pkl.load(open(args.cite_submission_path, 'rb'))
    
    
    print('merge submission')
    multi_submission.iloc[:len(cite_submission.ravel())] = cite_submission.ravel()
    assert not multi_submission.isna().any()
    multi_submission = multi_submission.round(6) # reduce the size of the csv
    print('save submission')
    multi_submission.to_csv(args.save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cite_submission_path', default='/home/jxf/code/kaggle_MSCI/results/0903_mlp_1024_submission_citeseq.pkl', type=str)
    parser.add_argument('--multi_submission_path', default='/home/jxf/code/kaggle_MSCI/results/0903_mlp_1024_partial_submission_multiome.pkl', type=str)
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/results/0903_submission.csv', type=str)
    args = parser.parse_args()

    main(args)