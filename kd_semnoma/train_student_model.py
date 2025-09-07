import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.SemNOMA_with_KD import SemNOMA_KD_AWGN, SemNOMA_KD_Rayleigh
#from datasets.cifar10_group_wrapper import train_loader, valid_loader
from datasets.ffhq256_group_wrapper import train_loader, valid_loader
from models.losses import Sum_MSE, Sum_MAE, PSNR


def resume_from_ckpt(ckpt_path, model, optimizer=None, lr_scheduler=None):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    print(f"loaded checkpoint {ckpt_path}")
    print(f"model was trained for {ckpt['epoch']} epochs")
    return ckpt["epoch"]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Wait for how many epochs after the last validation loss improvement
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            If True, print a message for each validation loss improvement
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Minimum change in the monitored quantity to qualify as an improvement
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
 
    def __call__(self, val_loss, model):
 
        score = -val_loss
 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
 
    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        Save model when validation loss decreases.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth') # This will store the best model parameters so far
        # torch.save(model, 'finish_model.pkl') # This will store the best model so far
        self.val_loss_min = val_loss


def train_hkd_feat_mse(net, EPOCH, train_loader, valid_loader, checkpoint_path='checkpoint_HKD_M_32_SA_weight100_mae10_teacherM32_adamw_A3_mae_2user_idx1_crosskd_mae_Rayleigh_FFHQ256_59.pth', teacher_checkpoint_path='/home/patrick/wqf/SeqNetHKD/DeepJSCC_NOMA/checkpoint_PerfectSIC_A1_M_16_mult_1_no_bind_mae_embed_1decoders_2user_noawgn.pth'):
    # Optimizer, AdamW + weight_decay
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Loss function
    criterion = Sum_MAE().cuda()
    #criterion = Sum_MSE().cuda()
    psnr = PSNR().cuda()
    
    # Define best loss
    best_loss = 200
    
    # Training
    writer = SummaryWriter('./path_2users/to/log_monitor_HKD_M_32_SA_weight100_mae10_teacherM32_A3_mae_2user_idx1_crosskd_mae_Rayleigh_FFHQ256_59')
    early_stopping = EarlyStopping(patience=10, verbose=True)
    '''
    # Load teacher_head pretrained model
    if os.path.exists(teacher_checkpoint_path):
        print("Loading teacher_head pretrained model...")
        resume_from_ckpt(teacher_checkpoint_path, net.teacher_head)
        for param in net.teacher_head.parameters():
            param.requires_grad = False
        net.teacher_head.eval()
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")
    '''
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('best_loss', 200)
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("Checkpoint structure is not as expected. Starting from scratch.")
            start_epoch = 0
    else:
        print("Starting from scratch")
        start_epoch = 0

    accumulation_steps = 1 
    
    for epoch in range(start_epoch, EPOCH):
        # Training metrics
        train_metrics = {'total_loss': 0, 'mse': 0, 'psnr': 0, 'kd_spatial': 0, 'kd_cross': 0}
        num_train = 0
        # Validation metrics
        val_metrics = {'mse': 0, 'psnr': 0}
        num_val = 0

        start = datetime.datetime.now()
        print('Epoch', epoch)
        
        # Training loop
        net.train()
        for step, b_x in tqdm(enumerate(train_loader), total=len(train_loader)):
            num_train += 1
            image, csi = b_x
            image, csi = image.cuda(), csi.cuda()

            output, loss_dict = net(image, csi)  
            loss_mse = criterion(output, image)
            t_psnr = psnr(output, image)
            
            # Total loss calculation
            total_loss = 10 * loss_mse + sum(loss_dict.values())

            # Backward pass
            total_loss.backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Update training metrics
            train_metrics['total_loss'] += total_loss.item()
            train_metrics['mse'] += loss_mse.item()
            train_metrics['psnr'] += t_psnr.item()
            for loss_name, loss_value in loss_dict.items():
                train_metrics[loss_name] += loss_value.item()

        # Calculate average training metrics
        for metric in train_metrics:
            train_metrics[metric] /= num_train

        # Validation loop
        with torch.no_grad():
            net.eval()
            for step, b_x in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                num_val += 1
                image, csi = b_x
                image, csi = image.cuda(), csi.cuda()
                output = net(image, csi)
                loss_v = criterion(output, image)
                v_psnr = psnr(output, image)
                
                val_metrics['mse'] += loss_v.item()
                val_metrics['psnr'] += v_psnr.item()

        # Calculate average validation metrics
        for metric in val_metrics:
            val_metrics[metric] /= num_val

        time0 = datetime.datetime.now() - start
        
        # Print epoch results
        print(f'Epoch: {epoch}, time: {time0}')
        print(f'Train - Total Loss: {train_metrics["total_loss"]:.8f}, MSE: {train_metrics["mse"]:.8f}, PSNR: {train_metrics["psnr"]:.8f}')
        print(f'KD Losses - kd_spatial: {train_metrics["kd_spatial"]:.8f}, kd_cross: {train_metrics["kd_cross"]:.8f}')
        print(f'Val - MSE: {val_metrics["mse"]:.8f}, PSNR: {val_metrics["psnr"]:.8f}')

        # Write to tensorboard
        writer.add_scalar('Train/Total_Loss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Train/MSE', train_metrics['mse'], epoch)
        writer.add_scalar('Train/PSNR', train_metrics['psnr'], epoch)
        writer.add_scalar('Train/KD_SA', train_metrics['kd_spatial'], epoch)
        writer.add_scalar('Train/KD_CROSS', train_metrics['kd_cross'], epoch)
        writer.add_scalar('Validation/MSE', val_metrics['mse'], epoch)
        writer.add_scalar('Validation/PSNR', val_metrics['psnr'], epoch)

        # Early stopping check
        early_stopping(val_metrics['mse'], net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        # Save best model
        if val_metrics['mse'] < best_loss:
            best_loss = val_metrics['mse']
            torch.save(net.state_dict(), 'test_comp0_UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mse_crosskd_idx1_Rayleigh_FFHQ256_59.pth')
            print('model saved')

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

    writer.close()
    print(f'The best MSE is: {best_loss:.3f}')


EPOCH = 1000
num_devices = 2
M = 32
net_kd = SemNOMA_KD_AWGN(num_devices, M, teacher_checkpoint_path='UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=6).cuda()
#net_kd = SemNOMA_KD_Rayleigh(num_devices, M, teacher_checkpoint_path='UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=1).cuda()
train_hkd_feat_mse(net_kd, EPOCH, train_loader, valid_loader)
