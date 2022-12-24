import torch.nn as nn
import torch
import numpy as np

class param_gen_2dlaplace_simple(nn.Module):
    def __init__(self, n_alphas):
        super(param_gen_2dlaplace_simple, self).__init__()
        self.n_alphas = n_alphas

        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.leaky_relu = nn.LeakyReLU()
        self.para_a = nn.Linear(in_features=752, out_features=n_alphas)
        self.para_b = nn.Linear(in_features=752, out_features=n_alphas)
        self.para_A = nn.Linear(in_features=752, out_features=n_alphas)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input_r, t1_in, t2_in, a_range=[], b_range=[]):
        x = self.conv(input_r)
        x = self.leaky_relu(x).view(1, -1)
        par_a = self.para_a(x)
        par_b = self.para_b(x)
        par_A = self.para_A(x)

        # Ak = (torch.cos(par_A) + 1.0).view(1, self.n_alphas)
        Ak = self.sigmoid1(par_A).view(1, self.n_alphas)

        z = (
            self.range_cal(a_range, par_a)
            if a_range
            else self.sigmoid(par_a).view(self.n_alphas, 1)
        )
        y = (
            self.range_cal(b_range, par_b)
            if b_range
            else self.sigmoid(par_b).view(self.n_alphas, 1)
        )
        a = -torch.log(z)
        b = -torch.log(y)
        output = harmonic_gen_2dlaplace_z(z, y, Ak, t1_in, t2_in)

        return a, b, Ak, output

    def range_cal(self, arg0, arg1):
        z_min = torch.exp(-arg0[1])
        z_max = torch.exp(-arg0[0])
        return (self.sigmoid(arg1) * (z_max - z_min) + z_min).view(self.n_alphas, 1)

class spec_gen_2dlaplace_simple(nn.Module):
    def __init__(self, n_dim_x, n_dim_y):
        super(spec_gen_2dlaplace_simple, self).__init__()
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y

        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.leaky_relu = nn.LeakyReLU()
        self.para_A = nn.Linear(in_features=752, out_features=n_dim_x*n_dim_y)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_r, Kx, Ky):
        x = self.conv(input_r)
        x = self.leaky_relu(x).view(1, -1)
        par_A = self.para_A(x)

        Ak = self.sigmoid(par_A).view(self.n_dim_y, self.n_dim_x)

        output_x = torch.matmul(Ky, Ak)
        output = torch.matmul(output_x, Kx.permute(1, 0))

        return Ak, output

def harmonic_gen_2dlaplace_z(z, y, Ak, t1, t2):
    '''
    size of z: [n_decay, 1]
    size of Ak: [n_freq, n_decay]
    size of t: [1, n_grad]
    size of output: [n_freq, n_grad]
    '''
    N = Ak.size(1)
    C1 = torch.pow(z, t1)
    C2 = torch.pow(y, t2)
    output = torch.zeros(size=[t1.size(1), t2.size(1)], device=C1.device)
    for i in np.arange(N):
        output = output + Ak[:, i] * torch.matmul(C1[i, :].unsqueeze(1), C2[i, :].unsqueeze(0))

    return output