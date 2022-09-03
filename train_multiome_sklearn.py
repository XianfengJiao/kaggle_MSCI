import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from argparse import ArgumentParser
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

from utils.metric_utils import correlation_score
from utils.data_utils import PreprocessCiteseq, PreprocessMultiome

DATA_DIR = "/home/jxf/code/kaggle_MSCI/input/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


def load_multi_data(start, stop):
    print('Start Processing Multiome Data')
    preprocessor = PreprocessMultiome()
    multi_train_x = preprocessor.fit_transform(pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=stop).values)

    multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=stop)
    y_columns = multi_train_y.columns
    multi_train_y = multi_train_y.values
    print('Finish Processing Multiome Data')
    
    return multi_train_x, multi_train_y, y_columns, preprocessor

def load_metadata():
    print('Start Loading Metadata')
    df_cell = pd.read_csv(FP_CELL_METADATA)
    df_cell_cite = df_cell[df_cell.technology=="citeseq"]
    df_cell_multi = df_cell[df_cell.technology=="multiome"]
    print('metadata-citeseq:', df_cell_cite.shape,'\n','metadata-multiome:', df_cell_multi.shape)
    print('Finish Loading Metadata')
    return df_cell_cite, df_cell_multi

def prepare_submission(y_columns):
    eval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col='row_id')
    # Convert the string columns to more efficient categorical types
    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())
    # Create the set of needed cell_ids
    cell_id_set = set(eval_ids.cell_id)
    # Convert the string gene_ids to a more efficient categorical dtype
    y_columns = pd.CategoricalIndex(y_columns, dtype=eval_ids.gene_id.dtype, name='gene_id')
    # Prepare an empty series which will be filled with predictions
    submission = pd.Series(name='target',
                        index=pd.MultiIndex.from_frame(eval_ids), 
                        dtype=np.float32)
    return submission, cell_id_set, eval_ids

def gen_submission(y_columns, model, preprocessor, chunksize=5000):
    submission, cell_id_set, eval_ids = prepare_submission(y_columns)
    # Process the test data in chunks of 5000 rows
    start = 0
    total_rows = 0
    while True:
        multi_test_x = None # Free the memory if necessary
        gc.collect()
        # Read the 5000 rows and select the 30 % subset which is needed for the submission
        multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS, start=start, stop=start+chunksize)
        rows_read = len(multi_test_x)
        needed_row_mask = multi_test_x.index.isin(cell_id_set)
        multi_test_x = multi_test_x.loc[needed_row_mask]
        
        # Keep the index (the cell_ids) for later
        multi_test_index = multi_test_x.index
        
        # Predict
        multi_test_x = multi_test_x.values
        multi_test_x = preprocessor.transform(multi_test_x)
        test_pred = model.predict(multi_test_x)
        
        # Convert the predictions to a dataframe so that they can be matched with eval_ids
        test_pred = pd.DataFrame(test_pred,
                                index=pd.CategoricalIndex(multi_test_index,
                                                        dtype=eval_ids.cell_id.dtype,
                                                        name='cell_id'),
                                columns=y_columns)
        gc.collect()
        
        # Fill the predictions into the submission series row by row
        for i, (index, row) in enumerate(test_pred.iterrows()):
            row = row.reindex(eval_ids.gene_id[eval_ids.cell_id == index])
            submission.loc[index] = row.values
        print('na:', submission.isna().sum())

        #test_pred_list.append(test_pred)
        total_rows += len(multi_test_x)
        print(total_rows)
        if rows_read < chunksize: break # this was the last chunk
        start += chunksize
        
    del multi_test_x, multi_test_index, needed_row_mask
    submission.reset_index(drop=True, inplace=True)
    submission.index.name = 'row_id'
    return submission

def main(args):
    model = Ridge(copy_X=False) # we overwrite the training data
    multi_train_x, multi_train_y, y_columns, preprocessor = load_multi_data(0, 6000)
    print('Start Training Model for Multiome Data')
    model.fit(multi_train_x, multi_train_y)
    print('Finish Training Model for Multiome Data')
    del multi_train_x, multi_train_y # free the RAM
    gc.collect()
    submission = gen_submission(y_columns, model, preprocessor, chunksize=5000)
    with open(args.save_path, 'wb') as f: pickle.dump(submission, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/results/0831_partial_submission_multi.pkl', type=str)
    args = parser.parse_args()

    main(args)