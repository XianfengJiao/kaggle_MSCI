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
from utils.data_utils import PreprocessCiteseq

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

def load_cite_data():
    print('Start Processing Citeseq Data')
    preprocessor = PreprocessCiteseq()
    cite_train_x = preprocessor.fit_transform(pd.read_hdf(FP_CITE_TRAIN_INPUTS).values)
    cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
    cite_test_x = preprocessor.transform(pd.read_hdf(FP_CITE_TEST_INPUTS).values)
    print('Finish Processing Citeseq Data')
    
    return cite_train_x, cite_train_y, cite_test_x

def main(args):
    model = Ridge(copy_X=False) # we overwrite the training data
    cite_train_x, cite_train_y, cite_test_x = load_cite_data()
    print('Start Training Model for Citeseq Data')
    model.fit(cite_train_x, cite_train_y)
    print('Finish Training Model for Citeseq Data')
    del cite_train_x, cite_train_y
    gc.collect()
    test_pred = model.predict(cite_test_x)
    with open(args.save_path, 'wb') as f: pickle.dump(test_pred, f) # float32 array of shape (48663, 140)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_path', default='/home/jxf/code/kaggle_MSCI/results/0831_submission_citeseq.pkl', type=str)
    args = parser.parse_args()
    main(args)
    