import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
sys.path.append("..")
from utils.data_utils import PreprocessCiteseq

class CiteseqDataset(Dataset):
    def __init__(self, input_data, data_path, type, kfold=0, preprocessor=None, scaler=None, input_label=None, is_train=False):
        
        if preprocessor:
            prefix = preprocessor.get_name()+'_'+scaler.__class__.__name__
            preprocessed_path = os.path.join(data_path, prefix, 'kfold-'+str(kfold), type+'_cite_preprocessed_data.h5')
        else:
            preprocessed_path = os.path.join(data_path, 'ori_cite', 'kfold-'+str(kfold), type+'_cite_ori.h5')
        
        if not os.path.isfile(preprocessed_path):
            # -------------------------- Preprocessing DATA --------------------------
            # input_data = pd.read_hdf(input_path).values
            os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
            print('#'*20,'Start Processing Data for fold', kfold,'#'*20)
            if preprocessor:
                if not is_train:
                    input_preprocessed = preprocessor.transform(input_data)
                else:
                    input_preprocessed = preprocessor.fit_transform(input_data)
                    pkl.dump(preprocessor, open(os.path.join(data_path, preprocessor.get_name()+'.pkl'), 'wb'))
            else:
                input_preprocessed = input_data
            
            if scaler:
                if is_train:
                    scaler.fit(input_preprocessed)
                input_preprocessed = scaler.transform(input_preprocessed)
            
            print('#'*20,'Finish Processing Data for fold', kfold,'#'*20)
            self._save_h5(preprocessed_path, input_preprocessed)
        else:
            # -------------------------- Load Preprocessed DATA --------------------------
            input_preprocessed = pd.read_hdf(preprocessed_path).values
        
        input_preprocessed = torch.tensor(
            input_preprocessed, dtype=torch.float32,
        )
        
        if type != 'test':
            # input_label = pd.read_hdf(label_path).values
            input_label = torch.tensor(
                input_label, dtype=torch.float32,
            )
            self.data = [(x, y) for x, y in zip(input_preprocessed, input_label)]
        else:
            print("Found preprocessed data for fold {}. Loading that!".format(kfold))
            self.data = input_preprocessed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _save_h5(self, save_path, data):
        h5 = pd.HDFStore(save_path,'w', complevel=4, complib='blosc')
        h5['values'] = pd.DataFrame(data)
        h5.close()
        