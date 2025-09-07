import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

from utils_eval import util_image
from utils_eval import util_common
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cyclegan_networks as cycnet
from Acnet_gan import Acnet_GAN   
from kd_semnoma.models.SemNOMA_with_KD import SemNOMA_KD_AWGN, SemNOMA_KD_Rayleigh
from kd_semnoma.datasets.ffhq256_group_wrapper import test_loader
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='EVAL_DEREFLECTION')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument(
            "-i",
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
parser.add_argument(
            "-o",
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
args = parser.parse_args()


def load_model(args):
    '''
    Load two-stage model, stage 1 A1 and stage 2 net_G
    '''
    # Load stage 1 model A1
    A1 = SemNOMA_KD_Rayleigh(num_devices=2, M=32, teacher_checkpoint_path=r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=4).to(device)
    checkpoint_A1 = torch.load(r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mse_crosskd_idx4_FFHQ256.pth')
    #A1 = SingleModelNet_A3_HKD(num_devices=2, M=32, teacher_checkpoint_path=r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_FFHQ256.pth', cross_layer_idx=4).to(device)
    #checkpoint_A1 = torch.load(r'D:\wqf\DifFace-master\pretrain_stage1_models\UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mae_CrossKD_teacher_idx4_AWGN_FFHQ256.pth')  # Load model parameters
    A1.load_state_dict(checkpoint_A1)
    A1.eval()
    
    # Load stage 2 generator net_G
    net_G = cycnet.define_G(input_nc=3, output_nc=3, ngf=64, netG=args.net_G,use_dropout=False,norm='none').to(device)
    print('loading the best checkpoint...')
    checkpoint = torch.load(os.path.join(args.ckptdir, 'last_ckpt.pt'))
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return A1, net_G



def run_eval(in_path,
            out_path
            ):
    '''
    Run evaluation of two-stage model, save images for subsequent metric evaluation
    '''

    print('running evaluation...')
    
    # File paths
    # Prepare result path, convert to Path objects
    in_path = in_path if isinstance(in_path, Path) else Path(in_path)
    out_path = out_path if isinstance(out_path, Path) else Path(out_path)
    restored_face_dir = out_path / 'restored_faces'  # Path to save restored images
    stage1_dir = out_path / 'stage1_outputs'  # Path to save Stage 1 outputs
    gt_dir = out_path / 'ground_truth'  # Path to save ground truth images
    
    # Create directories explicitly
    util_common.mkdir(out_path, parents=True)
    util_common.mkdir(restored_face_dir, parents=True)
    util_common.mkdir(stage1_dir, parents=True)  
    util_common.mkdir(gt_dir, parents=True)   

    # Load dataset
    dataloader = test_loader[15]   # Dataset test at -5dB
    
    for batch_idx, data in enumerate(tqdm(dataloader)):
        y0, csi = data
        y0, csi = y0.to(device), csi.to(device)
        batchsize, num_devices, channels, height, width = y0.shape
                                       
        # Save ground truth images
        y0_flat = y0.reshape(-1, channels, height, width)  # (batchsize * num_devices, c, h, w)
        for jj in range(y0_flat.shape[0]):
            gt_img = y0_flat[jj].permute(1, 2, 0).cpu().numpy()  # h x w x c, [0,1], RGB
            save_path = gt_dir / f'batch_{batch_idx}_{jj}_gt.png'
            util_image.imwrite(gt_img, save_path, chn='rgb', dtype_in='float32')

        # Through stage 1 and 2 and save images
        # Through stage 1 network   
        with torch.no_grad():     
            stage1_out = A1(y0, csi)   #(batch*num_devices,16,8,8)
            
            stage1_out.clamp_(0.0, 1.0)
            # Through stage 2 network, use initial estimation as condition
            # Through generator
            sample = net_G(stage1_out)
            
            sample.clamp_(0.0, 1.0)

            # Save Stage 1 outputs
            for jj in range(stage1_out.shape[0]):
                stage1_img = stage1_out[jj].permute(1, 2, 0).cpu().numpy()
                save_path = stage1_dir / f'batch_{batch_idx}_{jj}_stage1.png'
                util_image.imwrite(stage1_img, save_path, chn='rgb', dtype_in='float32')

            # Save generator restored images
            for jj in range(sample.shape[0]):
                restored_face = sample[jj].permute(1, 2, 0).cpu().numpy()  # h x w x c, [0,1], RGB
                save_path = restored_face_dir / f'batch_{batch_idx}_{jj}_restored.png'
                util_image.imwrite(restored_face, save_path, chn='rgb', dtype_in='float32')


if __name__ == '__main__':

    # args.dataset = 'dogsflowers'
    # args.net_G = 'unet_128'
    # args.in_size = 128
    # args.ckptdir = 'checkpoints'

    # args.dataset = 'mnist'
    # args.net_G = 'unet_64'
    # args.in_size = 64
    # args.ckptdir = 'checkpoints'

    args.save_output = True

    A1, net_G = load_model(args)
    run_eval(in_path=args.in_path,
             out_path=args.out_path)





