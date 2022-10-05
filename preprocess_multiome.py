import os
import pandas as pd
import numpy as np
import scipy.sparse
import scipy
from argparse import ArgumentParser


def convert_to_parquet(filename, out_filename):
    df = pd.read_csv(filename)
    df.to_parquet(out_filename + ".parquet")
    
def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start+chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize: 
            del df_chunk
            break
        del df_chunk
        start += chunksize
        
    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list
    
    all_indices = np.hstack(chunks_index_list)
    
    scipy.sparse.save_npz(out_filename+"_values.sparse", all_data_sparse)
    np.savez(out_filename+"_idxcol.npz", index=all_indices, columns =columns_name)
    
    
def main(args):
    # convert_h5_to_sparse_csr(args.train_input_path, os.path.join(args.save_path, "train_multi_inputs"))
    # convert_h5_to_sparse_csr(args.test_input_path, os.path.join(args.save_path, "test_multi_inputs"))
    convert_h5_to_sparse_csr(args.train_target_path, os.path.join(args.save_path, "train_multi_targets"))
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_input_path', default='/home/jxf/code/kaggle_MSCI/input/train_multi_inputs.h5', type=str)
    parser.add_argument('--test_input_path', default='/home/jxf/code/kaggle_MSCI/input/test_multi_inputs.h5', type=str)
    parser.add_argument('--train_target_path', default='/home/jxf/code/kaggle_MSCI/input/train_multi_targets.h5', type=str)
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/input/', type=str)
    args = parser.parse_args()
    main(args)


