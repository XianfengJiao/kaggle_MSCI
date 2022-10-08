import os
import torch
import torch.nn as nn
import torch.utils.data as data
from shutil import copyfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from datasets import CiteseqDataset
from networks import MLP
from trainer import Citeseq_Trainer
from utils.data_utils import PreprocessCiteseq, PreprocessCiteseqWithExpert, setup_seed


def train(args):
    pass




def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ------------------------ Instantiate Dataset ------------------------
    # preprocessor = PreprocessCiteseq(args.n_components)
    important_cols_index = pkl.load(open(args.important_cols_index_path, 'rb'))
    
    print('#'*20,'Start Loading Data','#'*20)
    constant_cols = pkl.load(open(args.constant_cols_path, 'rb'))
    input_data = pd.read_hdf(args.train_input_path).drop(columns=constant_cols).values
    input_label = pd.read_hdf(args.train_target_path).values
    test_input_data = pd.read_hdf(args.test_input_path).drop(columns=constant_cols).values
    print('#'*20,'End Loading Data','#'*20)
    
    if args.test:
        preprocessor = pkl.load(open(args.preprocessor_path, 'rb'))
        model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
        model.load_state_dict(torch.load(args.pretrain_model_path))
        trainer = Citeseq_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            log_dir=args.log_dir,
            lr=args.lr,
            device=device,
            model=model,
            early_stop=args.early_stop,
            save_path=args.ckpt_save_path
        )
        test_pred = trainer.inference(model, test_loader)
        with open(args.submission_save_path, 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
        exit(0)
        
    # length=len(input_data)
    # train_size,val_size = int(0.8*length),length-int(0.8*length)
    # train_set,val_set=data.random_split(input_data,[train_size,val_size])
    kfold_cor = []
    kfold_loss = []
    # ------------------------ kfold ------------------------
    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for i, (train_set, val_set) in enumerate(kfold.split(input_data, input_label)):
        preprocessor = PreprocessCiteseqWithExpert(args.n_components, important_cols_index)
        scaler = StandardScaler()
        
        train_data = CiteseqDataset(input_data=input_data[train_set], 
                                    type='train',
                                    scaler=scaler,
                                    kfold=i,
                                    data_path=os.path.dirname(args.train_input_path), 
                                    preprocessor=preprocessor, 
                                    input_label = input_label[train_set], 
                                    is_train = True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        
        valid_data = CiteseqDataset(input_data=input_data[val_set], 
                                    type='valid',
                                    scaler=scaler,
                                    kfold=i,
                                    data_path=os.path.dirname(args.train_input_path), 
                                    preprocessor=preprocessor, 
                                    input_label = input_label[val_set])
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        
        test_data = CiteseqDataset(input_data = test_input_data, 
                                type='test',
                                scaler=scaler,
                                kfold=i,
                                data_path=os.path.dirname(args.test_input_path), 
                                preprocessor=preprocessor)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        # ----------------- Instantiate Model --------------------------
        model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    
        # ----------------- Instantiate Trainer ------------------------
        trainer = Citeseq_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            log_dir=os.path.join(args.log_dir, 'kfold-'+str(i)),
            lr=args.lr,
            device=device,
            model=model,
            early_stop=args.early_stop,
            save_path=os.path.join(args.ckpt_save_path, 'kfold-'+str(i))
        )
        
        print('#'*20,"Starting training...",'#'*20)
        for epoch in range(1, args.epoch + 1):
            trainer.train_epoch(epoch)
            if trainer.remain_step == 0:
                break
        print('#'*20,"End training fold", i,'#'*20)
        print('Eval best corr:', trainer.best_cor)
        print('Eval best loss:', trainer.best_loss)
        kfold_cor.append(trainer.best_cor)
        kfold_loss.append(trainer.best_loss)
        
        copyfile(trainer.best_loss_model_path, trainer.best_loss_model_path.replace('.pth', '_'+str(trainer.best_loss)+'.pth'))
        copyfile(trainer.best_corr_model_path, trainer.best_corr_model_path.replace('.pth', '_'+str(trainer.best_cor)+'.pth'))
        print('#'*20,"Starting testing...",'#'*20)
        model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
        model.load_state_dict(torch.load(trainer.best_corr_model_path))
        test_pred = trainer.inference(model, test_loader)
        os.makedirs(os.path.join(args.submission_save_path, 'kfold-'+str(i)), exist_ok=True)
        with open(os.path.join(args.submission_save_path, 'kfold-'+str(i),'submission_citeseq.pkl'), 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
        print('#'*20,"End testing fold", i,'#'*20)
    
    print('#'*20,"End testing in all fold",'#'*20)
    print("Loss: {:.4f} ({:.4f})".format(np.mean(kfold_loss), np.std(kfold_loss)))
    print("Corr: {:.4f} ({:.4f})".format(np.mean(kfold_cor), np.std(kfold_cor)))
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--submission_save_path', default='/home/jxf/code/kaggle_MSCI/results/debug_mlp_1024', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/kaggle_MSCI/checkpoints/debug_MLP_1024', type=str)
    parser.add_argument('--log_dir', default='/home/jxf/code/kaggle_MSCI/logs/debug_MLP_1024', type=str)
    parser.add_argument("--test", action='store_true', help="only to test.",)
    parser.add_argument("--pretrain_model_path", default='', type=str)
    parser.add_argument('--preprocessor_path', default='', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--input_dim', default=60+144, type=int)
    parser.add_argument('--early_stop', default=15, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_components', default=60, type=int)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--output_dim', default=140, type=int)
    parser.add_argument('--train_input_path', default='/home/jxf/code/kaggle_MSCI/input/train_cite_inputs.h5', type=str)
    parser.add_argument('--test_input_path', default='/home/jxf/code/kaggle_MSCI/input/test_cite_inputs.h5', type=str)
    parser.add_argument('--train_target_path', default='/home/jxf/code/kaggle_MSCI/input/train_cite_targets.h5', type=str)
    parser.add_argument('--constant_cols_path', default='/home/jxf/code/kaggle_MSCI/input/constant_cols.pkl', type=str)
    parser.add_argument('--important_cols_index_path', default='/home/jxf/code/kaggle_MSCI/input/important_cols_index.pkl', type=str)
    args = parser.parse_args()
    main(args)