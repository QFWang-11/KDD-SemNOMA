import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import datetime
from models.losses import Sum_MSE, Sum_MAE, MSE, Average_PSNR, PSNR, SSIM, MS_SSIM
from models.baseline import SingleModelNet
from models.SemNOMA import SemNOMA_Rayleigh
from models.SemNOMA_with_KD import SemNOMA_KD_AWGN, SemNOMA_KD_Rayleigh
from models.Teacher_Model import Teacher_Model_AWGN, Teacher_Model_Rayleigh
from datasets.ffhq256_group_wrapper import test_loader
#from datasets.cifar10_group_wrapper import test_loader


def test_different_snr(net, test_loader):
    # load trained model 
    net.load_state_dict(torch.load('UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mse_crosskd_idx4_FFHQ256.pth'))  #load模型参数

    #loss function
    #criterion = Sum_MSE().cuda()
    criterion = Sum_MAE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()
    # testing
    Loss = []
    Psnr = []
    Ssim = []
    #print(len(test_loader))
    with torch.no_grad():
        for test_dataloader in test_loader:     # test under different SNR [0, 20] dB
            num_test = 0 
            test_loss = 0
            test_psnr = 0
            test_ssim = 0
            for step, b_x in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                num_test = num_test + 1
                image, csi = b_x
                image, csi = image.cuda(), csi.cuda()
                net.eval()
                output = net(image, csi) 
                loss_test = criterion(output, image)
                psnr_test = psnr(output, image)
                ssim_test = ssim(output, image)
                test_psnr = test_psnr + psnr_test.item()
                test_ssim = test_ssim + ssim_test.item()
                test_loss = test_loss + loss_test.item()

            test_loss = test_loss / num_test 
            test_psnr = test_psnr / num_test
            test_ssim = test_ssim / num_test
            Loss.append(test_loss)
            Psnr.append(test_psnr)
            Ssim.append(test_ssim)

            print("\n# Loss listas")
            print("Loss = [")
            for loss in Loss:
                print(f"    {loss:.8f}, ")
            print("]")
            
            print("\n# PSNR listas")
            print("Psnr = [")
            for psnr_val in Psnr:
                print(f"    {psnr_val:.8f}, ")
            print("]")

            print("\n# SSIM listas")
            print("Ssim = [")
            for ssim_val in Ssim:
                print(f"    {ssim_val:.8f}, ")
            print("]")
            #print('SNR:%.8f'% snr, 'test loss:%.8f'% test_loss, 'test_psnr:%.8f'% test_psnr)


def test_specific_snr(net, test_loader):
    # load trained model 
    net.load_state_dict(torch.load('UL_NOMA_2_Users_A3_M_32_HKD_SA_weight100_mae10_teacherM32_mse_crosskd_idx3_Rayleigh_FFHQ256_59.pth'))  #load模型参数

    #loss function
    #criterion = Sum_MSE().cuda()
    criterion = Sum_MAE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()
    # testing
    Loss = []
    Psnr = []
    Ssim = []
    #snr = -1
    #print(len(test_loader))
    with torch.no_grad():
            test_dataloader = test_loader[10]     # test under one specific SNR
            #snr = snr + 1
            num_test = 0 
            test_loss = 0
            test_psnr = 0
            test_ssim = 0
            for step, b_x in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                num_test = num_test + 1
                image, csi = b_x
                image, csi = image.cuda(), csi.cuda()
                net.eval()
                output = net(image, csi) 
                loss_test = criterion(output, image)
                psnr_test = psnr(output, image)
                ssim_test = ssim(output, image)
                test_psnr = test_psnr + psnr_test.item()
                test_ssim = test_ssim + ssim_test.item()
                test_loss = test_loss + loss_test.item()

            test_loss = test_loss / num_test 
            test_psnr = test_psnr / num_test
            test_ssim = test_ssim / num_test
            Loss.append(test_loss)
            Psnr.append(test_psnr)
            Ssim.append(test_ssim)
            print("\n# Loss list")
            print("Loss = [")
            for loss in Loss:
                print(f"    {loss:.8f}, ")
            print("]")
            
            print("\n# PSNR list")
            print("Psnr = [")
            for psnr_val in Psnr:
                print(f"    {psnr_val:.8f}, ")
            print("]")

            print("\n# SSIM list")
            print("Ssim = [")
            for ssim_val in Ssim:
                print(f"    {ssim_val:.8f}, ")
            print("]")
            #print('SNR:%.8f'% snr, 'test loss:%.8f'% test_loss, 'test_psnr:%.8f'% test_psnr)


num_devices = 2
M = 32
#net_baseline = SingleModelNet(num_devices=num_devices, M=M).cuda()
#net_teacher = Teacher_Model_Rayleigh(num_devices=num_devices, M=M).cuda()
#net_semnoma = SemNOMA_Rayleigh(num_devices=num_devices, M=M).cuda()
net_kd_semnoma = SemNOMA_KD_Rayleigh(num_devices=num_devices, M=M, teacher_checkpoint_path='UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=3).cuda()
test_specific_snr(net_kd_semnoma, test_loader)