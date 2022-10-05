import os
import pandas as pd
import numpy as np
import scipy.sparse
import scipy
from argparse import ArgumentParser


def split(path, split_size, save_path):
    os.makedirs(save_path, exist_ok=True)
    data = pd.read_hdf(path, start=0, stop=split_size)
    fn = os.path.basename(path)
    store = pd.HDFStore(os.path.join(save_path, fn.replace('.h5', '_splited_6000.h5')))
    store.put(key='data', value=data)
    store.close()
    print(f'split {path} successful')
    # test = pd.read_hdf(os.path.join(save_path, fn.replace('.h5', '_splited_6000.h5')), key='data')
    # pass
    

def main(args):
    split(args.train_multi_inputs, args.split_size, args.save_path)
    split(args.test_multi_inputs, args.split_size, args.save_path)
    split(args.train_multi_targets, args.split_size, args.save_path)
    split(args.train_cite_inputs, args.split_size, args.save_path)
    split(args.test_cite_inputs, args.split_size, args.save_path)
    split(args.train_cite_targets, args.split_size, args.save_path)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_multi_inputs', default='/home/jxf/code/kaggle_MSCI/input/train_multi_inputs.h5', type=str)
    parser.add_argument('--test_multi_inputs', default='/home/jxf/code/kaggle_MSCI/input/test_multi_inputs.h5', type=str)
    parser.add_argument('--train_multi_targets', default='/home/jxf/code/kaggle_MSCI/input/train_multi_targets.h5', type=str)
    parser.add_argument('--train_cite_inputs', default='/home/jxf/code/kaggle_MSCI/input/train_cite_inputs.h5', type=str)
    parser.add_argument('--test_cite_inputs', default='/home/jxf/code/kaggle_MSCI/input/test_cite_inputs.h5', type=str)
    parser.add_argument('--train_cite_targets', default='/home/jxf/code/kaggle_MSCI/input/train_cite_targets.h5', type=str)
    parser.add_argument('--split_size', default=6000, type=int)
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/input/splited/', type=str)
    args = parser.parse_args()
    main(args)