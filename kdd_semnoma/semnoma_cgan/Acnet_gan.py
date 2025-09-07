import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import loss
import cyclegan_networks as cycnet

import torch
torch.cuda.current_device()
import torchvision.models as torchmodels
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from kd_semnoma.models.SemNOMA_with_KD import SemNOMA_KD_AWGN, SemNOMA_KD_Rayleigh
from vgg16 import vgg
import torch.nn.functional as F
import pyiqa

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ALPHA_MAX = 0.6
ALPHA_MIN = 0.4

# Two-stage: establishing mapping from coarse estimation images to images

class Acnet_GAN():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        # Definition of generator, discriminator and pretrained models
        #self.A1 = SingleModelNet_A3_HKD(num_devices=2, M=32, teacher_checkpoint_path=r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_FFHQ256.pth', cross_layer_idx=4).to(device)
        self.A1 = SemNOMA_KD_Rayleigh(num_devices=2, M=32, teacher_checkpoint_path=r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=4).to(device)
        self.vgg16 = vgg.to(device)
        self.net_D = cycnet.define_D(input_nc=3, ndf=64, netD='n_layers', n_layers_D=3).to(device)  # Concatenate condition with image(3+3)
        self.net_G = cycnet.define_G(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.net_G,use_dropout=False,norm='none' ).to(device)
        #self.convtrans = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4, padding=0).to(device)
        
        # Initialize image quality assessment
        self.ssim_metric = pyiqa.create_metric('ssim', device=device)
        self.lpips_metric = pyiqa.create_metric('lpips', device=device)
        self.fid_metric = pyiqa.create_metric('fid', device=device)
        
        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        # Define optimizers
        # define optimizers
        self.optimizer_G = optim.Adam(
            self.net_G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(
            self.net_D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)
    
        # coefficient to balance loss functions
        self.lambda_L1 = args.lambda_L1
        self.lambda_adv = args.lambda_adv
        self.lambda_fea = args.lambda_fea

        # based on which metric to update the "best" ckpt
        self.metric = args.metric

        # define some other vars to record the training states
        self.running_psnr = []  # New addition: store PSNR
        self.running_acc = []  # Used to store combined_loss
        self.running_ssim = []
        self.running_fid = []
        self.running_lpips = []  # Replace original running_lpips to use pyiqa's LPIPS
        
        self.epoch_acc = 0
        self.best_val_acc = 1e9  # for mse, rmse, a lower score is better
        self.num_devices = 2
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_num_epochs
        self.g_pred = None
        self.x_stage_1 = None
        self.y_upsample = None
        #self.G_pred1 = None
        #self.G_pred2 = None
        self.batch = None
        self.G_loss = None
        self.D_loss = None
        self.Enc_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.D_fake_pool_1 = utils.ImagePool(pool_size=50)
        #self.D1_fake_pool = utils.ImagePool(pool_size=50)
        #self.D2_fake_pool = utils.ImagePool(pool_size=50)
        #self.D3_fake_pool = utils.ImagePool(pool_size=50)

        # define the loss functions
        '''
        if args.pixel_loss == 'minimum_pixel_loss':
            self._pxl_loss = loss.MinimumPixelLoss(opt=1) # 1 for L1 and 2 for L2
        elif args.pixel_loss == 'pixel_loss':
            self._pxl_loss = loss.PixelLoss(opt=1)  # 1 for L1 and 2 for L2
        else:
            raise NotImplementedError('pixel loss function [%s] is not implemented', args.pixel_loss)
        '''
        self._pxl_loss = loss.Sum_MAE()
        self._gan_loss = loss.GANLoss(gan_mode='wgangp').to(device)   
        self._exclusion_loss = loss.ExclusionLoss()
        self._kurtosis_loss = loss.KurtosisLoss()
        self._feature_loss = loss.FeatureLoss()
        self._psnr_acc = loss.PSNR()
        # enable some losses?
        self.with_d1d2 = args.enable_d1d2
        self.with_d3 = args.enable_d3
        self.with_exclusion_loss = args.enable_exclusion_loss
        self.with_kurtosis_loss = args.enable_kurtosis_loss

        # m-th epoch to activate adversarial training
        self.m_epoch_activate_adv = int(self.max_num_epochs / 20) + 1   #11

        # output auto-enhancement?
        self.output_auto_enhance = args.output_auto_enhance

        # use synfake to train D?
        self.synfake = args.enable_synfake

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # visualize model
        if args.print_models:
            self._visualize_models()

    # Model visualization
    def _visualize_models(self):

        from torchviz import make_dot

        # visualize models with the package torchviz
        y = self.net_G(torch.rand(4, 3, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_G.named_parameters()))
        mygraph.render('G')
        y = self.net_D1(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D1.named_parameters()))
        mygraph.render('D1')
        y = self.net_D2(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D2.named_parameters()))
        mygraph.render('D2')
        y = self.net_D3(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D3.named_parameters()))
        mygraph.render('D3')

    # Model saving and loading
    def _load_checkpoint(self):
        # Load pretrained weights of stage 1 network
        #self.A1.load_state_dict(torch.load(r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mae_CrossKD_teacher_idx4_AWGN_FFHQ256.pth'))  #load model parameters
        self.A1.load_state_dict(torch.load(r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mse_crosskd_idx4_FFHQ256.pth'))
        self.A1.eval()
        
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update net_D states
            self.net_D.load_state_dict(checkpoint['model_D_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.exp_lr_scheduler_D.load_state_dict(
                checkpoint['exp_lr_scheduler_D_state_dict'])
            self.net_D.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d, %s)' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id, self.metric))
            print()

        else:
            print('training from scratch...')


    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'model_D_state_dict': self.net_D.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'exp_lr_scheduler_D_state_dict': self.exp_lr_scheduler_D.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))


    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
        self.exp_lr_scheduler_D.step()


    def _compute_acc(self):

        output1 = self.g_pred
        image1 = self.image
        
        # Normalize to [0, 1]
        def normalize_to_01(tensor):
            # If value range is not in [0, 1], normalize
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val < 0 or max_val > 1:
                # Linear mapping from [min_val, max_val] to [0, 1]
                tensor = (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
            return tensor.clamp(0, 1)  # Ensure values are within [0, 1]

        output1 = normalize_to_01(output1)
        image1 = normalize_to_01(image1)

        # Calculate PSNR
        psnr = self._psnr_acc(output1, image1)

        # Calculate SSIM
        ssim = self.ssim_metric(output1, image1).mean()

        # Calculate LPIPS
        lpips = self.lpips_metric(output1, image1).mean()
        
        # Calculate pixel_loss and feature_loss
        pixel_loss = self._pxl_loss(output1, image1)
        feature_loss = self._feature_loss(output1, image1)
        combined_loss = self.lambda_L1 * pixel_loss + self.lambda_fea * feature_loss
        

        return psnr, ssim, lpips, combined_loss


    def _collect_running_batch_states(self):
        # Get PSNR, SSIM, LPIPS
        psnr, ssim, lpips, combined_loss = self._compute_acc()
        self.running_psnr.append(psnr.item())
        self.running_ssim.append(ssim.item())
        self.running_lpips.append(lpips.item())
        self.running_acc.append(combined_loss.item())

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])
            
        # Print metrics every 100 batches or at the end of epoch
        if np.mod(self.batch_id, 100) == 1 or self.batch_id == m-1:
            print('Is_training: %s. [%d/%d][%d/%d], G_loss: %.8f, D_loss: %.8f, '
              'running_PSNR: %.8f, running_SSIM: %.8f, running_LPIPS: %.8f, running_combined_loss: %.8f, running_FID: %.8f (%s)' 
              % (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                 self.G_loss.item(), self.D_loss.item(),
                 np.nanmean(self.running_psnr), np.nanmean(self.running_ssim),
                 np.nanmean(self.running_lpips), np.nanmean(self.running_acc), np.nanmean(self.running_fid), self.metric))

        if np.mod(self.batch_id, 1000) == 1 or self.batch_id == m-1:
            vis_input = utils.make_numpy_grid(self.image)
            vis_pred = utils.make_numpy_grid(self.g_pred)
            if self.output_auto_enhance:
                vis_pred1 = vis_pred1*1.5
                vis_pred2 = vis_pred2*1.5
            vis = np.concatenate([vis_input, vis_pred], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)



    def _collect_epoch_states(self):

        self.epoch_acc = np.mean(self.running_acc)   # For psnr
        print('Is_training: %s. Epoch %d / %d, epoch_acc= %.8f (%s),' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc, self.metric))
        print()


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        print()

        # update the best model
        if self.epoch_acc < self.best_val_acc:
            # a lower score is better
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            print('*' * 10 + 'Best model updated!')
            print()



    def _clear_cache(self):
        self.running_acc = []


    def _forward_pass(self, batch):
        self.batch = batch
        img_in_noma, csi = batch #(batch, num_devices, 3, 32, 32)  (batch,1)
        img_in_noma, csi = img_in_noma.to(device), csi.to(device)
        batchsize, num_devices, channels, height, width = img_in_noma.shape
        img_in = img_in_noma.view(batchsize*num_devices, channels, height, width)
        # Through stage 1 network
        self.image = img_in          
        x_stage1 = self.A1(img_in_noma, csi)   #(batch*num_devices,16,8,8)
        self.x_stage_1 = x_stage1
        # Through stage 2 network, use initial estimation as condition
        # Through generator
        self.g_pred = self.net_G(self.x_stage_1)             #pred(batch*num_devices,3,256,256) input(batch*num_devices,3,256,256), output(batch*num_devices,3,256,256)


    def _backward_D(self):
        self.D_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.epoch_id >= 1:       # Only train generator in early training to avoid discriminator providing too strong gradients to generator, making training difficult to stabilize
        #(batch,num_devices*16,8,8)

            #fake_cat = torch.cat((self.x_stage_1, self.g_pred), dim=1).detach()   # Coarse estimation features as conditional input
            fake_cat = self.g_pred.detach()
            fake_cat = self.D_fake_pool_1.query(fake_cat)                   # Store in buffer
            D_pred_fake = self.net_D(fake_cat)
            #real_cat = torch.cat((self.x_stage_1, self.image), dim=1).detach() 
            real_cat = self.image.detach() 
            D_pred_real = self.net_D(real_cat)
            D_adv_loss_fake = self._gan_loss(D_pred_fake, False)
            D_adv_loss_real = self._gan_loss(D_pred_real, True)
            D_adv_loss = 0.5*(D_adv_loss_fake + D_adv_loss_real)
            
            self.D_loss += self.lambda_adv*D_adv_loss

        self.D_loss.backward()


    def _backward_G(self):

        pixel_loss = self._pxl_loss.forward(
            self.g_pred, self.image
        )
        feature_loss = self._feature_loss.forward(
            self.g_pred, self.image
        )
        G_adv_loss = torch.tensor(0.0, requires_grad=True).to(device)
        exclusion_loss = torch.tensor(0.0, requires_grad=True).to(device)
        kurtosis_loss = torch.tensor(0.0, requires_grad=True).to(device)

        if self.with_exclusion_loss:
            exclusion_loss = self._exclusion_loss.forward(
                G_pred1=self.g_pred[:,0,...], G_pred2=self.g_pred[:,1,...])

        if self.with_kurtosis_loss:
            kurtosis_loss = self._kurtosis_loss.forward(
                G_pred1=self.G_pred1, G_pred2=self.G_pred2)
        # D1,D2 are conditional discriminators, input label and generated images for discriminator
        # D2 discriminates whether generated images are successfully separated
        if self.epoch_id >= 2:

            #fake_cat = torch.cat((self.x_stage_1, self.g_pred), dim=1)       # Mixed features and prediction features1 cat, conditional input sees both input condition and generation result, enhancing conditional generation capability
            fake_cat = self.g_pred.detach()
            #fake_cat = self.D_fake_pool_1.query(fake_cat)                   # Store in buffer
            D_pred_fake = self.net_D(fake_cat)
            D_adv_loss_fake = self._gan_loss(D_pred_fake, True)
            G_adv_loss += D_adv_loss_fake


        self.G_loss = self.lambda_L1*pixel_loss + \
                        self.lambda_L1*feature_loss + \
                        self.lambda_adv*G_adv_loss + \
                        2*exclusion_loss + \
                        kurtosis_loss
        
        self.G_loss.backward()


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            update_d_every = 100   # Update discriminator every 500 iterations
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.A1.eval()
            self.vgg16.eval()
            self.net_G.train()   # Set model to training mode
            self.net_D.train()  # Set model to training mode
            max_iterations_per_epoch = 10000  # Limit each epoch to 10000 iterations
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                #if self.batch_id >= max_iterations_per_epoch:
                #    break  # Early termination
                
                self._forward_pass(batch) #batch:(image,csi)
                # update D
                utils.set_requires_grad(self.net_D, True)
                self.optimizer_D.zero_grad()
                self._backward_D()        
                self.optimizer_D.step()
                # update G
                utils.set_requires_grad(self.net_D, False)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_schedulers()
            
            ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            # Set model to evaluate mode
            self.A1.eval()
            self.vgg16.eval()
            self.net_G.eval()
            self.net_D.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):  # self.batch updated
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()
