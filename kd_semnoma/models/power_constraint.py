import torch
import torch.nn as nn

class ComplexAveragePowerConstraint(nn.Module):
    
    def __init__(self, power, num_devices):
        super().__init__()
        
        self.power_factor = torch.sqrt(torch.tensor(power))                   # self.power_factor=1
        self.num_devices_factor = 1.0 / torch.sqrt(torch.tensor(num_devices)) # self.num_devices_factor=1/sqrt(2)

    def forward(self, hids, mult=1.0):
        hids_shape = hids.size()                 # (batch, num_devices, C_out, W_out, H_out) 
        #hids:(batch, num_devices, C_out, W_out, H_out)
        hids = hids.contiguous().view(hids_shape[0] * hids.shape[1], 2, -1) #(batch*num_devices,2,(C_out*W_out*H_out)/2)
        hids = torch.complex(hids[:, 0, :], hids[:, 1, :]) #cunstruct complex number
        #hids:(batch*num_devices, (C_out*W_out*H_out)/2)
        norm_factor = mult*torch.sqrt(1.0 / torch.tensor(hids_shape[1])) * self.power_factor * torch.sqrt(torch.tensor(hids.real.size(1), device=hids.device))
        # norm_factor = (sqrt(k/num_devices))
        hids = hids * torch.complex(norm_factor/torch.sqrt(torch.sum((hids * torch.conj(hids)).real, keepdims=True, dim=1)), torch.tensor(0.0, device=hids.device))
        # power* norm_factor * x / ||x||_2
        hids = torch.cat([hids.real, hids.imag], dim=1)
        hids = hids.view(hids_shape)

        return hids