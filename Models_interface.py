import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import utils, utils_torch
import MNIST_model

Models = {'mnist': MNIST_model,  
          'mnist_sup': MNIST_model,  
           'small-mnist':MNIST_model,
          }

def get_GDR_model(dataset, code_dim, noise_dim, label_dim = 0, supervised = True, label_training = True, D_sigmoid = True, Tanh = False, for_GP = False, **kwargs):
    
    dataset = dataset.lower()
    
    Model = Models[dataset]
    
    G = Model.Generator(code_dim = code_dim, noise_dim = noise_dim, label_dim = label_dim, Tanh = Tanh)
    if for_GP:
        D = Model.Discriminator_InsNorm(label_dim = label_dim, Sigmoid = D_sigmoid, out_label = supervised or label_training)
    else:
        D = Model.Discriminator(label_dim = label_dim, Sigmoid = D_sigmoid, out_label = supervised or label_training)
    R = Model.Regularizer(code_dim = code_dim)
    
    utils_torch.initialize_weights_xavier_normal(G, D, R)
    
    print('Model G D R for %s generated!' % dataset)
    print('code_dim: %d, noise_dim: %d, label_dim = %d, supervised = %s, D_sigmoid = %s, Tanh = %s' 
          % (code_dim, noise_dim, label_dim, supervised, D_sigmoid, Tanh) )
    print('Discriminator uses InstanceNormalization for Gradient Penalty' if for_GP else ' ')
    
    return G, D, R



def get_GER_model(dataset, code_dim, noise_dim, label_dim = 0, supervised = True, Tanh = False, **kwargs):
    
    dataset = dataset.lower()
    
    Model = Models[dataset]
    
    G = Model.Generator(code_dim = code_dim, noise_dim = noise_dim, label_dim = label_dim, Tanh = Tanh)
    E = Model.Encoder(code_dim = code_dim, noise_dim = noise_dim, label_dim = label_dim, out_label = supervised )
    R = Model.Regularizer(code_dim = code_dim)
    
    utils_torch.initialize_weights_xavier_normal(G, E, R)
    
    print('Model G E R for %s generated!' % dataset)
    print('code_dim: %d, noise_dim: %d, label_dim = %d, supervised = %s, Tanh = %s' 
          % (code_dim, noise_dim, label_dim, supervised, Tanh) )
    
    return G, E, R

