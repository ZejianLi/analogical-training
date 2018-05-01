import os, gzip, math
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def load_dataset(dataset, batch_size = 64, for_tanh = True):
    
    dataset = dataset.lower()
    mapping = {'mnist': load_dataset_MNIST,  
               'mnist_sup': load_dataset_MNIST,  
               'small-mnist': load_dataset_MNIST_small,
              }
    trainloader, testloader = mapping[dataset](batch_size, for_tanh = for_tanh)
    return trainloader, testloader



def load_dataset_MNIST_small(batch_size, for_tanh):
    trainloader, testloader = load_dataset_MNIST(batch_size = batch_size, num_train = 60, num_test = 10, for_tanh = for_tanh)
    print('MNIST small version loaded in normalized range ' + ('[-1, 1]' if for_tanh else '[0,1]'))
    return trainloader, testloader


def load_dataset_MNIST(batch_size = 64, download=True, num_train = 60000, num_test = 10000, for_tanh = True):
    """
    The output of torchvision datasets are PILImage images of range [0, 1].
    Transform them to Tensors of normalized range [-1, 1]
    """
    list_transforms = [ transforms.ToTensor() ]
    if for_tanh:
        list_transforms.append( transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) )
    transform = transforms.Compose( list_transforms )

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=download,
                                          transform=transform)
    
    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False,
                                         download=download,
                                         transform=transform)
    
    trainloader, testloader = get_data_loader(batch_size, trainset, testset, num_train, num_test)
    
    print('MNIST Data loaded in normalized range ' + ('[-1, 1]' if for_tanh else '[0,1]') )
    
    return trainloader, testloader


def get_data_loader(batch_size, trainset, testset, num_train=0, num_test=0):
    
    if num_train!=0 and num_test!=0:
        trainset.train_data = trainset.train_data[:num_train,:,:]
        trainset.train_labels = trainset.train_labels[:num_train]

        testset.test_data = testset.test_data[:num_test,:,:]
        testset.test_labels = testset.test_labels[:num_test]
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def long_tensor_to_onehot(idx, max_idx):
    return torch.zeros(idx.size()[0], max_idx).scatter_(1, idx.view(-1,1), 1).long()


def gen_random_labels(num_instance, max_idx):
    return torch.multinomial(torch.ones(max_idx), num_instance, replacement = True)


def gen_noise_Uniform(num_instance, n_dim=2, lower = 0, upper = 1):
    """generate n-dim uniform random noise"""
    return torch.rand(num_instance, n_dim)*(upper - lower) + lower


def gen_noise_Gaussian(num_instance, n_dim=2):
    """generate n-dim Gaussian random noise"""
    return torch.randn(num_instance, n_dim)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    

def print_network_num_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if hasattr(m.bias,'data'):
                m.bias.data.zero_()
    print('Weight initialized')

