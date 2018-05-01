import os, math
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pylab as plt
import utils


def to_var(x, gpu_mode = True, **kwargs):
    if torch.cuda.is_available() & gpu_mode:
        x = x.cuda()
    return Variable(x, **kwargs)


def to_np(x):
    if isinstance(x, Variable):
        x = x.data.cpu()
    elif not torch.is_tensor(x):
            raise TypeError('We need tensor here.')
    return x.cpu().numpy() 


def show_image(x, nrow = 16):
    
    if isinstance(x, Variable):
        x = x.cpu().data
    
    x = torchvision.utils.make_grid(x, nrow = nrow, padding = 4, scale_each=True, normalize = True) 
    x = to_np(x)
    
    if len(x.shape) == 4:
        x = x.transpose(0,2,3,1)
    elif len(x.shape) == 3:
        x = x.transpose(1,2,0)
    
    plt.imshow(x)
    plt.show()
 
    
def cov(x):
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1, keepdim = True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    return c
    
    
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout-in-tensorflow
# -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    
def block_conv_BN_IN_RELU(in_channel_size, out_channel_size, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, InsNorm = False, leaky = 0, dropout = False):
    model_list = []
    model_list.append( nn.Conv2d(
            in_channel_size, out_channel_size,
            kernel_size=kernel_size, stride=stride, padding=padding,) )
    
    if BatchNorm:
        model_list.append( nn.BatchNorm2d(out_channel_size) )
    elif InsNorm:
        model_list.append( nn.InstanceNorm2d(out_channel_size) )

    model_list.append( nn.ReLU() if leaky==0 else nn.LeakyReLU(leaky) )
    
    if dropout:
        model_list.append( nn.Dropout2d(0.2) )
    
    return nn.Sequential(*model_list)


def block_deconv_BN_IN_RELU(in_channel_size, out_channel_size, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, InsNorm = False, leaky = 0):
    model_list = []
    model_list.append( nn.ConvTranspose2d(
            in_channel_size, out_channel_size,
            kernel_size=kernel_size, stride=stride, padding=padding,) )
    
    if BatchNorm:
        model_list.append( nn.BatchNorm2d(out_channel_size) )
    elif InsNorm:
        model_list.append( nn.InstanceNorm2d(out_channel_size) )
    
    model_list.append( nn.ReLU() if leaky==0 else nn.LeakyReLU(leaky) )
    
    return nn.Sequential(*model_list)
    

def block_linear_BN_IN_RELU(in_size, out_size, BatchNorm = False, InsNorm = False, leaky = 0, dropout = False):
    
    model_list = []
    model_list.append( nn.Linear(in_size, out_size) )
    
    if BatchNorm:
        model_list.append( nn.BatchNorm1d(out_size) )
    elif InsNorm:
        model_list.append( nn.InstanceNorm1d(out_size) )
    
    model_list.append( nn.ReLU() if leaky==0 else nn.LeakyReLU(leaky) )
    
    if dropout:
        model_list.append( nn.Dropout(0.2) )
    
    return nn.Sequential(*model_list)
    


def block_conv_BN_RELU(in_channel_size, out_channel_size, kernel_size = 4, stride = 2, padding = 1, BatchNorm = True, leaky = 0, dropout = False):
    return block_conv_BN_IN_RELU(in_channel_size, out_channel_size, kernel_size, stride, padding, BatchNorm, False, leaky, dropout)

def block_deconv_BN_RELU(in_channel_size, out_channel_size, kernel_size = 4, stride = 2, padding = 1, BatchNorm = True, leaky = 0):
    return block_deconv_BN_IN_RELU(in_channel_size, out_channel_size, kernel_size, stride, padding, BatchNorm, False, leaky)

def block_linear_BN_RELU(in_size, out_size, BatchNorm = True, leaky = 0, dropout = False):
    return block_linear_BN_IN_RELU(in_size, out_size, BatchNorm, False, leaky, dropout)

    
    
def initialize_weights_xavier_normal(*nets):
    for net in nets:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
    print('Weight initialized with xavier_normal')
    