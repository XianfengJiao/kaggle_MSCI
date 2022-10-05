import os
import sys
import gc
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm
import pandas as pd
from torch import nn
import numpy as np
import scipy
sys.path.append("..")
from utils.metric_utils import correlation_score
from utils.data_utils import prepare_submission

class Multiome_Trainer(object):
    def __init__(
        self,
        train_loader,
        batch_size,
        num_epochs,
        log_dir,
        device,
        model,
        save_path,
        lr=1e-3,
        early_stop=15,
        loss='mse',
        checkpoint=None,
        valid_loader=None,
    ):
        self.train_loader = train_loader
        self.save_path = save_path
        self.model = model
        self.valid_loader = valid_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint = checkpoint
        self.log_dir = log_dir
        self.lr = lr
        self.best_loss_model_path = None
        self.best_corr_model_path = None
        self.loss_fn = self.configure_loss(loss)
        self.optimizer = self.configure_optimizer()
        self.device = device
        self.metric_fn = correlation_score
        self.early_stop = early_stop
        self.remain_step = self.early_stop
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.best_loss = 1e9
        self.best_cor = -1e9
        
    def train_epoch(self, epoch):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Epoch {}/{}".format(epoch, self.num_epochs), leave=False
        )
        
        loss_epoch = 0
        for x, y in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        
        loss_epoch /= len(self.train_loader)
        # Print epoch stats
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f}")
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)
        
        if loss_epoch < self.best_loss:
            self.best_loss = loss_epoch
            self.best_loss_model_path = os.path.join(self.save_path, 'best_loss.pth')
            torch.save(self.model.state_dict(), self.best_loss_model_path)
        
        if self.valid_loader:
            # Call evaluation
            eval_loss, eval_metric = self.evaluate(epoch)
            if eval_metric > self.best_cor:
                self.best_cor = eval_metric
                self.best_corr_model_path = os.path.join(self.save_path, 'best_corr.pth')
                torch.save(self.model.state_dict(), self.best_corr_model_path)
                self.remain_step = self.early_stop
            else:
                self.remain_step -= 1
    
    __call__ = train_epoch
    
    def evaluate(self, epoch):
        self.model.eval()
        eval_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        all_y = []
        all_pred = []
        with torch.no_grad():
            for x, y in eval_iterator:
                x = x.to(self.device)
                pred = self.model(x)
                all_y.append(y.numpy())
                all_pred.append(pred.detach().cpu().numpy())
        
        all_y = np.concatenate(all_y, axis=0).squeeze()
        all_pred = np.concatenate(all_pred, axis=0).squeeze()
        loss = mean_squared_error(all_pred, all_y)
        corr = self.metric_fn(all_pred, all_y)
        
        print(f"Epoch {epoch}:")
        print(f"Eval Loss: {loss:.4f}")
        print(f"Eval corr: {corr:.4f}")
        self.tensorwriter.add_scalar("eval_loss/epoch", loss, epoch)
        self.tensorwriter.add_scalar("eval_corr/epoch", corr, epoch)
        
        return loss, corr
    
    
    def _get_submission_sparse(self, test_pred, evaluation_ids_fp, y_columns, test_multi_inputs_idxcol_fp):
        eval_ids = pd.read_parquet(evaluation_ids_fp)
        eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
        eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())
        
        submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)
        # test_multi_inputs_idxcol_fp = '../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz'
        test_index = np.load(test_multi_inputs_idxcol_fp,
                    allow_pickle=True)["index"]
        
        cell_dict = dict((k,v) for v,k in enumerate(test_index)) 
        assert len(cell_dict)  == len(test_index)

        gene_dict = dict((k,v) for v,k in enumerate(y_columns))
        assert len(gene_dict) == len(y_columns)
        
        eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
        eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

        valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)
        submission.iloc[valid_multi_rows] = test_pred[eval_ids_cell_num[valid_multi_rows].to_numpy(),
                                                      eval_ids_gene_num[valid_multi_rows].to_numpy()]
        del eval_ids_cell_num, eval_ids_gene_num, valid_multi_rows, eval_ids, test_index, y_columns
        gc.collect()
        return submission
    
    def inference_sparse(self, model, evaluation_ids_fp, multiome_test_input_fp, y_columns, preprocessor, test_multi_inputs_idxcol_fp, scaler=None):
        # get prediction on test set
        model.eval()
        multi_test_x = scipy.sparse.load_npz(multiome_test_input_fp)
        multi_test_x = preprocessor.transform(multi_test_x)
        multi_test_x = torch.tensor(multi_test_x, dtype=torch.float32).to(self.device)
        test_pred = model(multi_test_x).detach().cpu().numpy()
        del multi_test_x
        gc.collect()
        
        submission = self._get_submission_sparse(test_pred, evaluation_ids_fp, y_columns, test_multi_inputs_idxcol_fp)
        return submission
        
        
    
    
    def inference(self, model, evaluation_ids_fp, multiome_test_input_fp, y_columns, preprocessor, scaler=None, chunksize=10000):
        model.eval()
        submission, cell_id_set, eval_ids = prepare_submission(y_columns, evaluation_ids_fp)
        start = 0
        total_rows = 0
        while True:
            multi_test_x = pd.read_hdf(multiome_test_input_fp, start=start, stop=start+chunksize)
            rows_read = len(multi_test_x)
            needed_row_mask = multi_test_x.index.isin(cell_id_set)
            multi_test_x = multi_test_x.loc[needed_row_mask]
            # Keep the index (the cell_ids) for later
            multi_test_index = multi_test_x.index
            multi_test_x = multi_test_x.values
            multi_test_x = preprocessor.transform(multi_test_x)
            if scaler:
                multi_test_x = scaler.transform(multi_test_x)
            multi_test_x = torch.tensor(multi_test_x, dtype=torch.float32).to(self.device)
            test_pred = model(multi_test_x).detach().cpu().numpy()
            test_pred = pd.DataFrame(test_pred,
                                    index=pd.CategoricalIndex(multi_test_index,
                                                            dtype=eval_ids.cell_id.dtype,
                                                            name='cell_id'),
                                    columns=y_columns)
            gc.collect()
            
            # Fill the predictions into the submission series row by row
            for (index, row) in tqdm(test_pred.iterrows()):
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
    
    def configure_loss(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError("Invalid Loss Type!")
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
