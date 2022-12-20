import torch.nn as nn
import torch
import numpy as np

class param_gen_dosy_simple(nn.Module):
    def __init__(self, n_alphas, n_peaks):
        super(param_gen_dosy_simple, self).__init__()
        self.n_alphas = n_alphas
        self.n_peaks = n_peaks

        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.leaky_relu = nn.LeakyReLU()
        self.para_a = nn.Linear(in_features=752, out_features=n_alphas)
        self.para_A = nn.Linear(in_features=752, out_features=n_alphas*n_peaks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_r, t_in, diff_range=[]):
        x = self.conv(input_r)
        x = self.leaky_relu(x).view(1, -1)
        par_a = self.para_a(x)
        par_A = self.para_A(x)

        Ak = (torch.cos(par_A) + 1.0).view(self.n_peaks, self.n_alphas)

        if not diff_range:
            z = self.sigmoid(par_a).view(self.n_alphas, 1)
            
        else:
            z_min = torch.exp(-diff_range[1])
            z_max = torch.exp(-diff_range[0])
            z = (self.sigmoid(par_a)*(z_max - z_min) + z_min).view(self.n_alphas, 1)
        
        a = -torch.log(z)
        output, C = harmonic_gen_dosy_z(z, Ak, t_in)
        norm_C = torch.square(torch.norm(C, p=2, dim=1, keepdim=True))

        return a, Ak, output, norm_C

def harmonic_gen_dosy_z(z, Ak, t):
    '''
    size of z: [n_decay, 1]
    size of Ak: [n_freq, n_decay]
    size of t: [1, n_grad]
    size of output: [n_freq, n_grad]
    '''
    
    C = torch.pow(z, t)
    output = torch.matmul(Ak, C)

    return output, C