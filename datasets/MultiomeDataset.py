import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
from torch.utils.data import Dataset
sys.path.append("..")
from utils.data_utils import PreprocessMultiome

class MultiomeDataset(Dataset):
    def __init__(self, input_path, preprocessor=None, scaler=None, label_path=None, is_train=False, ref_size=12000, chunk_size=5000, ):
        self.y_columns = None
        if preprocessor:
            prefix = preprocessor.get_name()+'_'+scaler.__class__.__name__
            preprocessed_path = input_path.replace('.h5', '_ref_size_' + str(ref_size) +'_preprocessed_' + prefix +'.h5')
        else:
            preprocessed_path = input_path
        
        if not os.path.isfile(preprocessed_path):
        # if True:
            # -------------------------- Preprocessing DATA --------------------------
            input_ref_data = pd.read_hdf(input_path, start=0, stop=ref_size).values
            input_ref_data = preprocessor.fit_transform(input_ref_data)
            if scaler:
                scaler.fit(input_ref_data)
                pkl.dump(scaler, open(os.path.join(os.path.dirname(preprocessed_path), 'ref_size_' + str(ref_size) + '_' + preprocessor.get_name()+'_'+scaler.__class__.__name__+'.pkl'), 'wb'))
            pkl.dump(preprocessor, open(os.path.join(os.path.dirname(preprocessed_path), 'ref_size_' + str(ref_size) + '_' + preprocessor.get_name()+'.pkl'), 'wb'))
            
            print('#'*20,'Start Processing Data','#'*20)
            start = 0
            total_rows = 0
            input_preprocessed = None
            while True:
                input_data = pd.read_hdf(input_path, start=start, stop=start+chunk_size).values
                rows_read = len(input_data)
                if preprocessor:
                    preprocessed_tmp = preprocessor.transform(input_data)
                else:
                    preprocessed_tmp = input_data
                
                if scaler:
                    preprocessed_tmp = scaler.transform(preprocessed_tmp)
                
                if start > 0:
                    input_preprocessed = np.vstack((input_preprocessed,preprocessed_tmp))
                else:
                    input_preprocessed = preprocessed_tmp
                
                total_rows += len(input_data)
                print(total_rows)
                gc.collect()
                if rows_read < chunk_size: break
                start += chunk_size
                
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
        