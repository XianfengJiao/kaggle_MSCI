import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
import scipy
from torch.utils.data import Dataset
sys.path.append("..")
from utils.data_utils import PreprocessMultiome

class MultiomeSparseDataset(Dataset):
    def __init__(self, input_path, preprocessor=None, label_path=None, is_train=False, ):
        self.y_columns = None
        # prefix = preprocessor.get_name()+'_'+scaler.__class__.__name__
        prefix = preprocessor.get_name()
        preprocessed_path = input_path.replace('.sparse.npz', '_preprocessed_' + prefix +'.h5')
        
        if not os.path.isfile(preprocessed_path):
        # if True:
            # -------------------------- Preprocessing DATA --------------------------
            print('#'*20,'Start Processing Data','#'*20)
            input_data = scipy.sparse.load_npz(input_path).astype('float16', copy=False)
            print('#'*20,'Load Data Successful','#'*20)
            input_preprocessed = preprocessor.fit_transform(input_data)
            print('#'*20,'TruncatedSVD Successful','#'*20)
            # if scaler:
            #     scaler.fit(input_data, with_mean=False)
            #     pkl.dump(scaler, open(os.path.join(os.path.dirname(preprocessed_path), preprocessor.get_name()+'_'+scaler.__class__.__name__+'.pkl'), 'wb'))
            pkl.dump(preprocessor, open(os.path.join(os.path.dirname(preprocessed_path), preprocessor.get_name()+'.pkl'), 'wb'))
            # input_preprocessed = scaler.transform(input_preprocessed)
            print('#'*20,'Finish Processing Data','#'*20)
            self._save_h5(preprocessed_path, input_preprocessed)
        else:
            # -------------------------- Load Preprocessed DATA --------------------------
            print("Found preprocessed data. Loading that!")
            input_preprocessed = pd.read_hdf(preprocessed_path).values
        
        input_preprocessed = torch.tensor(
            input_preprocessed, dtype=torch.float32,
        )
        
        if label_path:
            input_label = pd.read_hdf(label_path)
            self.y_columns = input_label.columns
            input_label = torch.tensor(
                input_label.values, dtype=torch.float32,
            )
            
            self.data = [(x, y) for x, y in zip(input_preprocessed, input_label)]
        else:
            self.data = input_preprocessed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _save_h5(self, save_path, data):
        h5 = pd.HDFStore(save_path,'w', complevel=4, complib='blosc')
        h5['values'] = pd.DataFrame(data)
        h5.close()
        