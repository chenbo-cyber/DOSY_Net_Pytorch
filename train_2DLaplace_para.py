import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
import util
from model_2DLaplace_est import *

DEVICE = torch.device("cuda:1")
logger = logging.getLogger(__name__)

def options():
    # Set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_type", default='Laplace2D', type=str, help='type of the dosy module')
    parser.add_argument("--learning_rate", default=1e-4, type=float, help='Learning rate at first')
    parser.add_argument("--input_file", default='data/simulation/testdataSigma0.015.mat', help='File path of the input data')
    parser.add_argument("--output_dir", default='Net_Results/', help='Output file path')
    parser.add_argument("--diff_range", default=[], type=float, help='The range of the diffusion coefficient')
    parser.add_argument("--n_decay", default=3, type=int, help='The number of different decay components')
    parser.add_argument("--dim_in", default=100, type=int, help='Dimension of the input of the network')
    parser.add_argument("--reg_A", default=0.01, type=float, help='Regularization parameter')
    parser.add_argument("--n_epochs", default=20000, type=int, help='Maximum iterations')
    parser.add_argument("--save_epoch", default=500, type=int, help='The step size to save the intermediate results')
    parser.add_argument("--fidelity", default='norm-2', help='Fidelity term setting, could be norm-1 or norm-2')
    parser.add_argument("--display", default=True, help='Do you want to show the convergence process in the terminal?')
    parser.add_argument("--no_cuda", action='store_true', help="avoid using CUDA when available")
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)
       
    args = parser.parse_args()            
    return args

def set_module(args, n_alphas):
    """
    Create a net module
    """
    net = None
    if args.module_type == 'Laplace2D':
        net = param_gen_2dlaplace_simple(n_alphas)
    else:
        raise NotImplementedError('module type not implemented')
    if args.use_cuda:
        net.to(DEVICE)
    return net

def train_laplace2d_parameters(args, laplace2d_module, laplace2d_optimizer, laplace2d_scheduler, train_input, train_label, t1, t2):
    """
    Train the dosy_parameters module for one epoch
    """
    laplace2d_module.train()
    total_loss = 0

    if args.use_cuda:
        input_r, label_output, b, t = train_input.to(DEVICE), train_label.to(DEVICE), t1.to(DEVICE), t2.to(DEVICE)
    laplace2d_optimizer.zero_grad()
    D_out, T2_out, Ak, X_output = laplace2d_module(input_r, b, t)
    
    # calculate loss
    err_tmp = label_output - X_output
    if args.fidelity == 'norm-2':
        fidelity_loss = torch.square(torch.sum(torch.pow(err_tmp, 2))).to(torch.float32)
    elif args.fidelity == 'norm-1':
        fidelity_loss = torch.sum(torch.abs(err_tmp)).to(torch.float32)
    else:
        raise NotImplementedError('Undefined fidelity term. Please input: norm-1 or norm-2 for "fidelity"')
    norm_label_tensor = torch.max(label_output, dim=1, keepdim=True)[0]
    total_loss = fidelity_loss

    total_loss.backward()
    laplace2d_optimizer.step()
    laplace2d_scheduler.step(total_loss)
    
    return D_out.reshape(-1).cpu().detach().numpy(), T2_out.reshape(-1).cpu().detach().numpy(), Ak.reshape(-1).cpu().detach().numpy(), total_loss, fidelity_loss

def train(args):
    # 0. set code running environment
    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    Folder_name = util.make_folder(args.output_path)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True

    # 1. load data & generate the input
    label_data, b, t = util.read_mat_laplace2d(args.input_file)
    input_r = np.random.randn(1,args.dim_in)
    input_r = np.expand_dims(input_r,axis=1)

    input_r = torch.from_numpy(input_r).float()
    label_output = torch.from_numpy(label_data).float()
    b = torch.from_numpy(b).float()
    t = torch.from_numpy(t).float()

    # 2. set training parameters
    laplace2d_module = set_module(args, n_alphas=args.n_decay)
    laplace2d_optimizer = torch.optim.Adam(laplace2d_module.parameters(), lr=args.learning_rate)
    laplace2d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(laplace2d_optimizer, 'min', patience=800, factor=0.8, verbose=True)
    
    epoch_start_time = time.time()
    epoch_ori_time = epoch_start_time
    min_total_loss = np.inf
    min_fidelity_loss = np.inf
    for epoch in range(1, args.n_epochs + 1):
        if epoch < args.n_epochs:
            D_save, T2_save, Ak_save, total_loss, fidelity_loss = train_laplace2d_parameters(args, laplace2d_module=laplace2d_module, laplace2d_optimizer=laplace2d_optimizer, laplace2d_scheduler=laplace2d_scheduler, train_input=input_r, train_label=label_output, t1=b, t2=t)

        # save dosy parameters and dosy module
        if epoch % args.save_epoch == 0 or epoch == args.n_epochs:
            logger.info("Epochs: %d / %d, Time: %.1f, total loss: %.6f, fidelity loss: %.6f", epoch, args.n_epochs, time.time() - epoch_start_time, total_loss, fidelity_loss)
            if epoch == args.n_epochs:
                logger.info("Loop over! Final epoch: %d, Total time cost: %.1f, final total loss: %.6f, final fidelity loss: %.6f", min_epoch, time.time() - epoch_ori_time, min_total_loss, min_fidelity_loss)
            epoch_start_time = time.time()

            if min_total_loss > total_loss:
                min_total_loss = total_loss
                min_fidelity_loss = fidelity_loss
                min_epoch = epoch
            
            util.save_param_laplace2d(D_save, T2_save, Ak_save, args.n_decay, Folder_name)
                # util.save(dosy_module, dosy_optimizer, dosy_scheduler, args, epoch, args.module_type)


if __name__ == "__main__":
    args = options()
    args.input_file = 'data/Laplace2D/laplace2D_net_input.mat'
    args.output_path = 'Net_Results/Laplace2D/'
    args.diff_range = []
    # args.diff_range = [3.0, 12.0]
    args.reg_A = 0.1
    args.n_epochs = 50000
    args.learning_rate = 1e-3
    args.fidelity = 'norm-2'
    args.n_decay = 2

    train(args)



