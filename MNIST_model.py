import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import utils, utils_torch
import torch.nn.functional as F


NUM_HIDDEN_1 = 1024
NUM_HIDDEN_2 = 128
NUM_CHANNELS_1 = 128
NUM_CHANNELS_2 = 64

class Generator(nn.Module):

    def __init__(self, code_dim = 4, noise_dim = 62, label_dim = 10, Tanh=False, **kwargs):
        super().__init__()
        self.fc = nn.Sequential(
            utils_torch.block_linear_BN_RELU(code_dim + noise_dim + label_dim, NUM_HIDDEN_1, leaky = 0), 
            utils_torch.block_linear_BN_RELU(NUM_HIDDEN_1, NUM_CHANNELS_1 * 49, leaky = 0)
            )
        
        self.dconv = nn.Sequential(
            utils_torch.block_deconv_BN_RELU(NUM_CHANNELS_1, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1, leaky = 0), 
            nn.ConvTranspose2d(in_channels = NUM_CHANNELS_2, out_channels = 1, kernel_size = 4, stride = 2, padding = 1) 
            )
        
        self.out = nn.Tanh() if Tanh else nn.Sigmoid()

    def forward(self, in_noise, in_code, in_label):
        in_concat = torch.cat([in_noise, in_code, in_label], 1)
        x = self.fc(in_concat)
        x = x.view(-1, NUM_CHANNELS_1, 7, 7)
        x = self.dconv(x)
        return self.out(x)
        
        

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, label_dim = 10, Sigmoid = True, **kwars):
        super(Discriminator, self).__init__()

        self.Sigmoid = Sigmoid
        if self.Sigmoid:
            self.sigmoid = nn.Sigmoid() # to indicate it sigmoid the output in __repr__

        self.conv = nn.Sequential(
            utils_torch.block_conv_BN_RELU(1, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, leaky = 0.2), 
            utils_torch.block_conv_BN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, kernel_size = 4, stride = 2, padding = 1, leaky = 0.2),
            )
        
        self.fc1 = utils_torch.block_linear_BN_RELU(NUM_CHANNELS_1 * 7 * 7, NUM_HIDDEN_1, leaky = 0.2)
        
        self.fc21 = nn.Linear(NUM_HIDDEN_1, 1)
        
        self.fc22 = nn.Sequential(
            utils_torch.block_linear_BN_RELU(NUM_HIDDEN_1, NUM_HIDDEN_2, leaky = 0.2),
            nn.Linear(NUM_HIDDEN_2, label_dim)
            )
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 7 * 7)
        out_fc1 = self.fc1(conv)
        out_D = self.fc21(out_fc1)
        if self.Sigmoid:
            out_D = F.sigmoid(out_D)
        out_label = self.fc22(out_fc1)
        return out_D, out_label



class Discriminator_InsNorm(nn.Module):
    """docstring for Discriminator_InsNorm"""
    def __init__(self, label_dim = 10, Sigmoid = False, **kwars):
        super(Discriminator_InsNorm, self).__init__()

        self.Sigmoid = Sigmoid
        if self.Sigmoid:
            self.sigmoid = nn.Sigmoid() # to indicate it sigmoid the output in __repr__

        self.conv = nn.Sequential(
            utils_torch.block_conv_BN_IN_RELU(1, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, InsNorm = False, leaky = 0.2), 
            utils_torch.block_conv_BN_IN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, InsNorm = True, leaky = 0.2),
            )
        
        self.fc1 = utils_torch.block_linear_BN_IN_RELU(NUM_CHANNELS_1 * 7 * 7, NUM_HIDDEN_1, leaky = 0.2, BatchNorm = True, InsNorm = True)
        
        self.fc21 = nn.Linear(NUM_HIDDEN_1, 1)
        
        self.fc22 = nn.Sequential(
            utils_torch.block_linear_BN_IN_RELU(NUM_HIDDEN_1, NUM_HIDDEN_2, leaky = 0.2, BatchNorm = True, InsNorm = True),
            nn.Linear(NUM_HIDDEN_2, label_dim)
            )
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 7 * 7)
        out_fc1 = self.fc1(conv)
        out_D = self.fc21(out_fc1)
        if self.Sigmoid:
            out_D = F.sigmoid(out_D)
        out_label = self.fc22(out_fc1)
        return out_D, out_label

    
    

class Regularizer(nn.Module):
    """docstring for Regularizer"""
    def __init__(self, code_dim = 10, in_channels = 1):
        super(Regularizer, self).__init__()

        self.conv = nn.Sequential(
            utils_torch.block_conv_BN_RELU(in_channels * 2, NUM_CHANNELS_2//2, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, leaky = 0.2, dropout = True),
            utils_torch.block_conv_BN_RELU(NUM_CHANNELS_2//2, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1, BatchNorm = True, leaky = 0.2, dropout = True),
        )
        self.fc = nn.Sequential(
            utils_torch.block_linear_BN_RELU(NUM_CHANNELS_2*7*7, NUM_HIDDEN_1, leaky = 0.2, BatchNorm = True, dropout = True),
            nn.Linear(NUM_HIDDEN_1, code_dim))

    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        x = x.view(-1, NUM_CHANNELS_2 * 7 * 7)
        out = self.fc(x)
        return out

    
    

class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self,  code_dim = 4, noise_dim = 62, label_dim = 10, **kwargs):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            utils_torch.block_conv_BN_RELU(1, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1, BatchNorm = False, leaky = 0.2), 
            utils_torch.block_conv_BN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, kernel_size = 4, stride = 2, padding = 1, leaky = 0.2),
            )
        
        self.fc1 = utils_torch.block_linear_BN_RELU(NUM_CHANNELS_1 * 7 * 7, NUM_HIDDEN_1, leaky = 0.2)
        
        self.fc21 = nn.Linear(NUM_HIDDEN_1, code_dim  * 2 )
        self.fc22 = nn.Linear(NUM_HIDDEN_1, noise_dim * 2 )
        
        self.fc23 = nn.Sequential(
            utils_torch.block_linear_BN_RELU(NUM_HIDDEN_1, NUM_HIDDEN_2, leaky = 0.2),
            nn.Linear(NUM_HIDDEN_2, label_dim)
            )
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 7 * 7)
        out_fc1 = self.fc1(conv)
        out_code = self.fc21(out_fc1)
        out_noise = self.fc22(out_fc1)
        out_label = self.fc23(out_fc1)
        return out_code, out_noise, out_label
    
    
