import os
import gc
import torch
import copy
import pandas as pd
from shutil import copyfile
import numpy as np
import torch.nn as nn
import pickle as pkl
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from datasets import MultiomeSparseDataset
from networks import MLP
from trainer import Multiome_Trainer
from utils.data_utils import PreprocessMultiomeWithTruncatedSVD, setup_seed

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    preprocessor = PreprocessMultiomeWithTruncatedSVD(args.n_components)
    if args.test:
        train_loader = None
        valid_loader = None
        
    else:
        train_data = MultiomeSparseDataset(input_path=args.train_input_path, 
                                     preprocessor=preprocessor, 
                                     label_path=args.train_target_path, 
                                     is_train=True)
        validation_split = .2
        shuffle_dataset = True
        random_seed= 42
        dataset_size = len(train_data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler)
    # test_data = MultiomeSparseDataset(input_path = args.test_input_path, preprocessor=preprocessor)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # ----------------- Instantiate Model --------------------------
    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    
    # ----------------- Instantiate Trainer ------------------------
    trainer = Multiome_Trainer(
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        log_dir=args.log_dir,
        device=device,
        model=model,
        lr=args.lr,
        early_stop=args.early_stop,
        save_path=args.ckpt_save_path
    )
    
    if args.test:
        print('#'*20, 'Prepare for Testing', '#'*20)
        y_columns = np.load(args.train_multi_targets_idxcol_path,
                   allow_pickle=True)["columns"]
        preprocessor = pkl.load(open(args.preprocessor_path, 'rb'))
        # scaler = pkl.load(open(args.scaler_path, 'rb'))
        model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
        model.load_state_dict(torch.load(args.pretrain_model_path))
        print('#'*20, 'Start Testing', '#'*20)
        test_pred = trainer.inference_sparse(model, multiome_test_input_fp=args.test_input_path, evaluation_ids_fp=args.evaluation_ids_path, y_columns=y_columns, preprocessor=preprocessor, test_multi_inputs_idxcol_fp=args.test_multi_inputs_idxcol_path)
        print('#'*20, 'End Testing', '#'*20)
        with open(args.submission_save_path, 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
        print('#'*20, 'Saving Submission Successful', '#'*20)
        exit(0)
    
    print('#'*20,"Starting training...",'#'*20)
    for epoch in range(1, args.epoch + 1):
        trainer.train_epoch(epoch)
        if trainer.remain_step == 0:
            break
    print('#'*20,"End training",'#'*20)
    print('Eval best corr:', trainer.best_cor)
    print('Eval best loss:', trainer.best_loss)
    
    copyfile(trainer.best_loss_model_path, trainer.best_loss_model_path.replace('.pth', '_'+str(trainer.best_loss)+'.pth'))
    copyfile(trainer.best_corr_model_path, trainer.best_corr_model_path.replace('.pth', '_'+str(trainer.best_cor)+'.pth'))
    
    print('#'*20,"Starting testing...",'#'*20)
    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    model.load_state_dict(torch.load(trainer.best_corr_model_path))
    y_columns = np.load(args.train_multi_targets_idxcol_path,
                   allow_pickle=True)["columns"]
    del train_data, train_loader # free the RAM
    gc.collect()
    test_pred = trainer.inference_sparse(model, multiome_test_input_fp=args.test_input_path, evaluation_ids_fp=args.evaluation_ids_path, y_columns=y_columns, preprocessor=preprocessor, test_multi_inputs_idxcol_fp=args.test_multi_inputs_idxcol_path)
    with open(args.submission_save_path, 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
    print('#'*20,"End testing",'#'*20)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--submission_save_path', default='/home/jxf/code/kaggle_MSCI/results/debug_mlp_1024_partial_submission_multiome.pkl', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/kaggle_MSCI/checkpoints/debug_Multiome_MLP_1024_b8192', type=str)
    parser.add_argument('--log_dir', default='/home/jxf/code/kaggle_MSCI/logs/debug_Multiome_MLP_1024_b8192', type=str)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument("--test", action='store_true', help="only to test.",)
    parser.add_argument("--pretrain_model_path", default='/home/jxf/code/kaggle_MSCI/checkpoints/Multiome_MLP_1024/best_loss1.8544642638701658.pth', type=str)
    parser.add_argument('--preprocessor_path', default='/home/jxf/code/kaggle_MSCI/input/PreprocessMultiome_1024.pkl', type=str)
    parser.add_argument('--scaler_path', default='', type=str)
    parser.add_argument('--input_dim', default=1024, type=int)
    parser.add_argument('--n_components', default=1024, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--early_stop', default=15, type=int)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)
    parser.add_argument('--output_dim', default=23418, type=int)
    parser.add_argument('--train_multi_targets_idxcol_path', default='/home/jxf/code/kaggle_MSCI/input/train_multi_targets_idxcol.npz', type=str)
    parser.add_argument('--test_multi_inputs_idxcol_path', default='/home/jxf/code/kaggle_MSCI/input/test_multi_inputs_idxcol.npz', type=str)
    parser.add_argument('--evaluation_ids_path', default='/home/jxf/code/kaggle_MSCI/input/evaluation.parquet', type=str)
    parser.add_argument('--train_input_path', default='/home/jxf/code/kaggle_MSCI/input/train_multi_inputs_values.sparse.npz', type=str)
    parser.add_argument('--test_input_path', default='/home/jxf/code/kaggle_MSCI/input/test_multi_inputs_values.sparse.npz', type=str)
    parser.add_argument('--train_target_path', default='/home/jxf/code/kaggle_MSCI/input/train_multi_targets.h5', type=str)
    args = parser.parse_args()
    main(args)