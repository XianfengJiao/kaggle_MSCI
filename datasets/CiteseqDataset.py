import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import Dataset
sys.path.append("..")
from utils.data_utils import PreprocessCiteseq

class CiteseqDataset(Dataset):
    def __init__(self, input_data, data_path, type, preprocessor=None, input_label=None, is_train=False):
        
        if preprocessor:
            prefix = preprocessor.get_name()
            preprocessed_path = os.path.join(data_path, type+'_cite_preprocessed_' + prefix +'.h5')
        else:
            preprocessed_path = os.path.join(data_path, type+'_cite_ori.h5')
        
        if not os.path.isfile(preprocessed_path):
            # -------------------------- Preprocessing DATA --------------------------
            # input_data = pd.read_hdf(input_path).values
            
            print('#'*20,'Start Processing Data','#'*20)
            if preprocessor:
                if not is_train:
                    input_preprocessed = preprocessor.transform(input_data)
                else:
                    input_preprocessed = preprocessor.fit_transform(input_data)
                    pkl.dump(preprocessor, open(os.path.join(data_path, preprocessor.get_name()+'.pkl'), 'wb'))
            else:
                input_preprocessed = input_data
            print('#'*20,'Finish Processing Data','#'*20)
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
            print("Found preprocessed data. Loading that!")
            self.data = input_preprocessed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _save_h5(self, save_path, data):
        h5 = pd.HDFStore(save_path,'w', complevel=4, complib='blosc')
        h5['values'] = pd.DataFrame(data)
        h5.close()
        