import os
import sys
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
sys.path.append("..")
from utils.metric_utils import correlation_score

class Citeseq_Trainer(object):
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
            
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.best_loss_model_path = os.path.join(self.save_path, 'best_loss.pth')
                torch.save(self.model.state_dict(), self.best_loss_model_path)
    
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
        print(f"Eval Loss: {loss:.4f} | Best Eval corr: {self.best_loss:.4f}")
        print(f"Eval corr: {corr:.4f} | Best Eval corr: {self.best_cor:.4f}")
        self.tensorwriter.add_scalar("eval_loss/epoch", loss, epoch)
        self.tensorwriter.add_scalar("eval_corr/epoch", corr, epoch)
        
        return loss, corr
        
    def inference(self, model, test_loader):
        model.eval()
        test_iterator = tqdm(
            test_loader, desc="Test", total=len(test_loader)
        )
        all_pred = []
        with torch.no_grad():
            for x in test_iterator:
                x = x.to(self.device)
                pred = self.model(x)
                all_pred.append(pred.detach().cpu())
        
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        return all_pred.numpy()
    
    def configure_loss(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError("Invalid Loss Type!")
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)