import os, gc
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
import scipy

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class PreprocessCiteseqWithExpert(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, important_cols_index) -> None:
        super().__init__()
        self.n_components = n_components
        self.important_cols_index = important_cols_index
    
    def get_name(self):
        return self.__class__.__name__ + '_' + str(self.n_components)
    
    def fit_transform(self, X):
        gc.collect()
        print(X.shape)
        X_imp = X[:, self.important_cols_index]
        self.pca = PCA(n_components=self.n_components, copy=False, random_state=1)
        X = self.pca.fit_transform(X)
        X = np.hstack([X, X_imp])
        print(X.shape)
        return X
    
    def transform(self, X):
        print(X.shape)
        gc.collect()
        X_imp = X[:, self.important_cols_index]
        X = self.pca.transform(X)
        X = np.hstack([X, X_imp])
        print(X.shape)
        return X

class PreprocessCiteseq(BaseEstimator, TransformerMixin):
    # columns_to_use = 12000
    columns_to_use = 0
    def __init__(self, n_components) -> None:
        super().__init__()
        self.n_components = n_components
    
    def get_name(self):
        return self.__class__.__name__ + '_' + str(self.n_components)
    
    @staticmethod
    def take_column_subset(X):
        if PreprocessCiteseq.columns_to_use > 0:
            return X[:,-PreprocessCiteseq.columns_to_use:]
        else:
            return X
    
    def transform(self, X):
        print(X.shape)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        X = self.pca.transform(X)
        print(X.shape)
        return X

    def fit_transform(self, X):
        gc.collect()
        print(X.shape)
        self.all_zero_columns = (X == 0).all(axis=0)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        self.pca = PCA(n_components=self.n_components, copy=False, random_state=1)
        X = self.pca.fit_transform(X)
        # plt.plot(self.pca.explained_variance_ratio_.cumsum())
        # plt.title("Cumulative explained variance ratio")
        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xlabel('PCA component')
        # plt.ylabel('Cumulative explained variance ratio')
        # plt.show()
        print(X.shape)
        return X


class PreprocessMultiomeWithTruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components) -> None:
        super().__init__()
        self.n_components = n_components
    
    def get_name(self):
        return self.__class__.__name__ + '_' + str(self.n_components)
    
    def fit_transform(self, X):
        print(X.shape)
        self.svd = TruncatedSVD(n_components = self.n_components, random_state = 1)
        X = self.svd.fit_transform(X)
        print(X.shape)
        return X
    
    def transform(self, X):
        print(X.shape)
        gc.collect()

        X = self.svd.transform(X)
        print(X.shape)
        return X



class PreprocessMultiome(BaseEstimator, TransformerMixin):
    # columns_to_use = slice(0, 14000)
    columns_to_use = None
    
    def __init__(self, n_components) -> None:
        super().__init__()
        self.n_components = n_components
        
    def get_name(self):
        return self.__class__.__name__ + '_' + str(self.n_components)
    
    @staticmethod
    def take_column_subset(X):
        if not PreprocessMultiome.columns_to_use:
            return X
        return X[:,PreprocessMultiome.columns_to_use]
    
    def transform(self, X):
        print(X.shape)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessMultiome.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        X = self.pca.transform(X)
        print(X.shape)
        return X

    def fit_transform(self, X):
        print(X.shape)
        self.all_zero_columns = (X == 0).all(axis=0)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessMultiome.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        self.pca = PCA(n_components=self.n_components, copy=False, random_state=1)
        X = self.pca.fit_transform(X)
        # plt.plot(self.pca.explained_variance_ratio_.cumsum())
        # plt.title("Cumulative explained variance ratio")
        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xlabel('PCA component')
        # plt.ylabel('Cumulative explained variance ratio')
        # plt.show()
        print(X.shape)
        return X

def prepare_submission(y_columns, evaluation_ids_fp):
    eval_ids = pd.read_csv(evaluation_ids_fp, index_col='row_id')
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