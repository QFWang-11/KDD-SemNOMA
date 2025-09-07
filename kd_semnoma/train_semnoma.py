import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import datetime
from models.losses import PSNR, Sum_MAE
from models.SemNOMA import SemNOMA_Rayleigh
#from datasets.cifar10_group_wrapper import train_loader, valid_loader
from datasets.ffhq256_group_wrapper import train_loader, valid_loader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
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
        Saves model when validation loss decrease.。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth') 
        self.val_loss_min = val_loss


def train(net, EPOCH, train_loader, valid_loader, checkpoint_path='checkpoint_A3_mae_M_16_opti_adamw_1_user_convnext_mae_Rayleigh_FFHQ256_OMA_57.pth'):
    # optimizer
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    # loss function
    #criterion = Sum_MSE().cuda()
    criterion = Sum_MAE().cuda()
    psnr = PSNR().cuda()
    
    # Define best loss
    best_loss = 200
    # Training
    writer = SummaryWriter('./path_2users/to/log_monitor_A3_mae_M_16_opti_adamw_1_user_convnext_mae_Rayleigh_FFHQ256_OMA_57')
    
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Check if the expected keys are in the checkpoint
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('best_loss', 200)  # Use default if best_loss not found
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("Checkpoint structure is not as expected. Starting from scratch.")
            start_epoch = 0
    else:
        print("Starting from scratch")
        start_epoch = 0

    accumulation_steps = 8 
    
    for epoch in range(start_epoch, EPOCH):
        loss_train = 0
        train_psnr = 0
        num_train = 0
        loss_val = 0
        val_psnr = 0
        num_val = 0

        start = datetime.datetime.now() 
        print('Epoch', epoch)
        lambda_sparse = 1e-4
        for step, b_x in tqdm(enumerate(train_loader), total=len(train_loader)):
            num_train += 1
            image, csi = b_x
            image, csi = image.cuda(), csi.cuda()

            net.train()  # self.training = True
            output = net(image, csi) 
            loss_t = criterion(output, image)
            #loss_teacher = criterion(teacher_output, image)
            #output1 = output.flatten(0, 1)
            #image1 = image.flatten(0, 1)
            t_psnr = psnr(output, image)
            train_psnr += t_psnr.item() 
            #loss_train = loss_t.item() + loss_teacher.item() + loss_kd.item()
            loss_train += loss_t.item()

            
            loss_t.backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()  
                optimizer.zero_grad() 

        loss_train /= num_train  
        train_psnr /= num_train  

        with torch.no_grad():
            net.eval()
            for step, b_x in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                num_val += 1
                image, csi = b_x
                image, csi = image.cuda(), csi.cuda()
                output = net(image, csi)
                loss_v = criterion(output, image)
                #output1 = output.flatten(0, 1)
                #image1 = image.flatten(0, 1)
                v_psnr = psnr(output, image)
                val_psnr += v_psnr.item()
                loss_val += loss_v.item()
            loss_val /= num_val
            val_psnr /= num_val
            time0 = datetime.datetime.now() - start
        print('Epoch:', epoch, 'time', time0, 'train loss:%.8f' % loss_train, 'train_psnr:%.8f' % train_psnr, 
              'val loss:%.8f' % loss_val, 'val_psnr:%.8f' % val_psnr)

        early_stopping(loss_val, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(net.state_dict(), 'UL_NOMA_1_Users_A3_M_16_convnext_mae_Rayleigh_FFHQ256_OMA_57.pth')
            print('model saved')

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

        writer.add_scalar('train_Loss', loss_train, epoch)
        writer.add_scalar('val_Loss', loss_val, epoch)
        writer.add_scalar('train_psnr', train_psnr, epoch)
        writer.add_scalar('val_psnr', val_psnr, epoch)
    writer.close()
    print('The best SE is: %.3f' % best_loss, 'dB')


num_devices = 2
M = 16
net = SemNOMA_Rayleigh(num_devices=num_devices, M=M).cuda()
EPOCH = 1000
train(net, EPOCH, train_loader, valid_loader)