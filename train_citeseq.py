import os
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from datasets import CiteseqDataset
from networks import MLP
from trainer import Citeseq_Trainer
from utils.data_utils import PreprocessCiteseq, setup_seed
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    preprocessor = PreprocessCiteseq(args.n_components)
    print('#'*20,'Start Loading Data','#'*20)
    input_data = pd.read_hdf(args.train_input_path).values
    input_label = pd.read_hdf(args.train_target_path).values
    test_input_data = pd.read_hdf(args.test_input_path).values
    print('#'*20,'End Loading Data','#'*20)
    length=len(input_data)
    train_size,val_size = int(0.8*length),length-int(0.8*length)
    train_set,val_set=data.random_split(input_data,[train_size,val_size])
    
    train_data = CiteseqDataset(input_data=input_data[train_set.indices], 
                                type='train',
                                data_path=os.path.dirname(args.train_input_path), 
                                preprocessor=preprocessor, 
                                input_label = input_label[train_set.indices], 
                                is_train = True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    valid_data = CiteseqDataset(input_data=input_data[val_set.indices], 
                                type='valid',
                                data_path=os.path.dirname(args.train_input_path), 
                                preprocessor=preprocessor, 
                                input_label = input_label[val_set.indices])
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    test_data = CiteseqDataset(input_data = test_input_data, 
                               type='test',
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
        log_dir=args.log_dir,
        lr=args.lr,
        device=device,
        model=model,
        save_path=args.ckpt_save_path
    )
    if args.test:
        preprocessor = pkl.load(open(args.preprocessor_path, 'rb'))
        model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
        model.load_state_dict(torch.load(args.pretrain_model_path))
        test_pred = trainer.inference(model, test_loader)
        with open(args.submission_save_path, 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
        exit(0)
    
    print('#'*20,"Starting training...",'#'*20)
    for epoch in range(1, args.epoch + 1):
        trainer.train_epoch(epoch)
    print('#'*20,"End training",'#'*20)
    print('#'*20,"Starting testing...",'#'*20)
    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    model.load_state_dict(torch.load(trainer.best_corr_model_path))
    test_pred = trainer.inference(model, test_loader)
    with open(args.submission_save_path, 'wb') as f: pkl.dump(test_pred, f) # float32 array of shape (48663, 140)
    print('#'*20,"End testing",'#'*20)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--submission_save_path', default='/home/jxf/code/kaggle_MSCI/results/debug_mlp_1024_submission_citeseq.pkl', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/kaggle_MSCI/checkpoints/debug_MLP_1024', type=str)
    parser.add_argument('--log_dir', default='/home/jxf/code/kaggle_MSCI/logs/debug_MLP_1024', type=str)
    parser.add_argument("--test", action='store_true', help="only to test.",)
    parser.add_argument("--pretrain_model_path", default='', type=str)
    parser.add_argument('--preprocessor_path', default='', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--input_dim', default=1024, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_components', default=1024, type=int)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--output_dim', default=140, type=int)
    parser.add_argument('--train_input_path', default='/home/jxf/code/kaggle_MSCI/input/train_cite_inputs.h5', type=str)
    parser.add_argument('--test_input_path', default='/home/jxf/code/kaggle_MSCI/input/test_cite_inputs.h5', type=str)
    parser.add_argument('--train_target_path', default='/home/jxf/code/kaggle_MSCI/input/train_cite_targets.h5', type=str)
    args = parser.parse_args()
    main(args)