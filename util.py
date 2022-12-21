# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import errno

def __clear_env():
    for key in list(globals().keys()):
        if not key.startswith("__"):
            globals().pop(key)


def read_mat_dosy(pathname):
    """
    Output: 
        label_data: DOSY spectral data with size [N_freq, N_grad]
        b_data: vector related to the gradients with size [1, N_grad]
    
    """
    read_data = scio.loadmat(str(pathname))
    label_data = read_data['S']
    b_data = read_data['b']
    return label_data, b_data

def read_mat_laplace2d(pathname):
    """
    Output: 
        label_data: 2D Laplace data with size [N_grad_b, N_grad_t]
        b_data: vector related to the gradients with size [1, N_grad_b]
        t_data: vector related to the gradients with size [1, N_grad_t]
    """
    read_data = scio.loadmat(str(pathname))
    label_data = read_data['S']
    b_data = read_data['b']
    t_data = read_data['t']
    return label_data, b_data, t_data

def make_folder(BaseDir):
    Name_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    subsubFolderName = str(Name_time)
    FolderName = '%s%s/' % (BaseDir,subsubFolderName)
    if not os.path.isdir(BaseDir):
        os.makedirs(FolderName)
    else:
        os.mkdir(FolderName)
    
    return FolderName

def save_csv(data, FolderName, file_name, shape, is_real):
    if is_real == 0:
        y2 = np.concatenate([np.squeeze(np.real(data)),np.squeeze(np.imag(data))])
        y2 = np.reshape(y2,[2,-1])
        df = pd.DataFrame(y2)
        df.to_csv(str(FolderName) + str(file_name),index_label='0rl\\1im')
    else:
        data = np.reshape(data, shape)
        df = pd.DataFrame(data)
        df.to_csv(str(FolderName) + str(file_name), index=0,header=0)
    
    
def save_param_dosy(aout1, Akout1, alpha_num, peak_num, FolderName):
    save_csv(aout1, FolderName, file_name='diffusion_coeffs.csv', shape=[-1,alpha_num], is_real=1)
    save_csv(Akout1, FolderName, file_name='Sp.csv', shape=[-1,peak_num * alpha_num], is_real=1)

def save_param_laplace2d(aout, bout, Ak, alpha_num, FolderName):
    save_csv(aout, FolderName, file_name='diffusion_coeffs.csv', shape=[-1,alpha_num], is_real=1)
    save_csv(bout, FolderName, file_name='relax_time.csv', shape=[-1,alpha_num], is_real=1)
    save_csv(Ak, FolderName, file_name='amplitude.csv', shape=[-1,alpha_num], is_real=1)


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def save(model, optimizer, scheduler, args, epoch, module_type):
    checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(os.path.join(args.output_dir, module_type)):
        os.makedirs(os.path.join(args.output_dir, module_type))
    cp = os.path.join(args.output_dir, module_type, 'last.pth')
    fn = os.path.join(args.output_dir, module_type, 'epoch_'+str(epoch)+'.pth')
    torch.save(checkpoint, fn)
    symlink_force(fn, cp)
