import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
import util
from model_DOSY_est import *

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

    return parser.parse_args()

def set_module(args, n_alphas, n_peaks):
    """
    Create a net module
    """
    net = None
    if args.module_type == 'DOSY':
        net = param_gen_dosy_simple(n_alphas, n_peaks)
    else:
        raise NotImplementedError('module type not implemented')
    if args.use_cuda:
        net.to(DEVICE)
    return net

def train_dosy_parameters(args, dosy_module, dosy_optimizer, dosy_scheduler, train_input, train_label, t):
    """
    Train the dosy_parameters module for one epoch
    """
    dosy_module.train()
    loss_dosy = 0

    if args.use_cuda:
        input_r, label_output, b = train_input.to(DEVICE), train_label.to(DEVICE), t.to(DEVICE)
    dosy_optimizer.zero_grad()
    dr_out, sp_out, X_output, normC_out = dosy_module(input_r, b)
    
    # calculate loss
    err_tmp = label_output - X_output
    if args.fidelity == 'norm-2':
        fidelity_loss = torch.square(torch.sum(torch.pow(err_tmp, 2))).to(torch.float32)
    elif args.fidelity == 'norm-1':
        fidelity_loss = torch.sum(torch.abs(err_tmp)).to(torch.float32)
    else:
        raise NotImplementedError('Undefined fidelity term. Please input: norm-1 or norm-2 for "fidelity"')
    norm_label_tensor = torch.max(label_output, dim=1, keepdim=True)[0]
    sp1_weighted = (sp_out * norm_label_tensor) / normC_out.T
    LW1_out = torch.sum(sp1_weighted)
    loss_dosy = fidelity_loss + args.reg_A * LW1_out

    loss_dosy.backward()
    dosy_optimizer.step()
    dosy_scheduler.step(loss_dosy)
    
    return dr_out.reshape(-1).cpu().detach().numpy(), sp_out.reshape(-1).cpu().detach().numpy(), loss_dosy, fidelity_loss

def train(args):
    # 0. set code running environment
    args.use_cuda = bool(torch.cuda.is_available() and not args.no_cuda)
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
    label_data0, b = util.read_mat_dosy(args.input_file)
    n_grad = b.shape[1]
    n_freq = label_data0.shape[0]
    input_r = np.random.randn(1,args.dim_in)
    input_r = np.expand_dims(input_r,axis=1)

    input_r = torch.from_numpy(input_r).float()
    label_output = torch.from_numpy(label_data0).float()
    b = torch.from_numpy(b).float()

    # 2. set training parameters
    dosy_module = set_module(args, n_alphas=args.n_decay, n_peaks=n_freq)
    dosy_optimizer = torch.optim.Adam(dosy_module.parameters(), lr=args.learning_rate)
    dosy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dosy_optimizer, 'min', patience=800, factor=0.8, verbose=True)

    epoch_start_time = time.time()
    epoch_ori_time = epoch_start_time
    min_total_loss = np.inf
    min_fidelity_loss = np.inf
    for epoch in range(1, args.n_epochs + 1):
        if epoch < args.n_epochs:
            dr_save, sp_save, total_loss, fidelity_loss = train_dosy_parameters(args, dosy_module=dosy_module, dosy_optimizer=dosy_optimizer, dosy_scheduler=dosy_scheduler, train_input=input_r, train_label=label_output, t=b)

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
                util.save_param_dosy(dr_save, sp_save, args.n_decay, n_freq, Folder_name)
                util.save(dosy_module, dosy_optimizer, dosy_scheduler, args, epoch, args.module_type)


if __name__ == "__main__":
    args = options()
    args.input_file = 'data/QGC/QGC_net_input.mat'
    args.output_path = 'Net_Results/QGC/'
    args.diff_range = []
    #args.diff_range = [3.0, 12.0]
    args.reg_A = 0.1
    args.n_epochs = 50000
    args.learning_rate = 1e-3
    args.fidelity = 'norm-2'
    args.n_decay = 3

    train(args)



