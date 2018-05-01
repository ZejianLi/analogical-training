import os, math, random, itertools, pickle, time, datetime
import imageio
from collections import deque
from tqdm import tnrange, tqdm_notebook, trange
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable, grad
import utils, utils_torch, Models_interface
import tensorboardX

torch.manual_seed(2333333)    # reproducible

from Basis import Basis

class ANA_GAN(Basis):
    def __init__(self, **kwargs): # parameters
        
        super(ANA_GAN, self).__init__(**kwargs)

        # whether to use WGAN
        self.WGAN = True
        self.WGAN_GP = True
        self.WGAN_num_critics = 3

        self.model_name = 'ANA_GAN_WGAN' if self.WGAN else 'ANA_GAN_GAN'

        print('INFO: using WGAN training num_critics = %d.' % self.WGAN_num_critics if self.WGAN else 'INFO: using GAN training.')
        if self.WGAN_GP:
            print('using WGAN-GP.')        

        self.save_path = os.path.join(self.save_dir, self.dataset, self.model_name)
        

        # get the models
        self.G, self.D, self.R = Models_interface.get_GDR_model(dataset = self.dataset,
            code_dim = self.code_dim, 
            noise_dim = self.noise_dim, 
            label_dim = self.num_labels, 
            supervised = self.supervised,
            label_training = self.label_training,
            D_sigmoid = False if self.WGAN else True, 
            Tanh = True,
            for_GP = True if self.WGAN and self.WGAN_GP else False)
        
        # optimiser, ADAM by default
        self.initializeOptimizaer_GDR()
        
    
        self.to_cuda()
        
        if self.gpu_mode:
            self.R.cuda()
    

    def initializeOptimizaer_GDR(self):
        
        if self.WGAN and not self.WGAN_GP:
            self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr, weight_decay = self.weight_decay)
            self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr, weight_decay = self.weight_decay)
            self.R_optimizer = optim.RMSprop(self.R.parameters(), lr=self.lr, weight_decay = self.weight_decay)
            print('Use RMSprop optimizers.')
        else:
            beta1, beta2 = 0.5, 0.99
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
            self.R_optimizer = optim.Adam(self.R.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
            print('Use ADAM optimizers.')
        
            
    def printNetArch(self):
        print('---------- Networks architecture -------------')
        print('G:')
        utils.print_network(self.G)
        print('D:')
        utils.print_network(self.D)
        print('R:')
        utils.print_network(self.R)
        print('-----------------------------------------------')

    def print_net_num_parameters(self):
        print('G:')
        utils.print_network_num_parameters(self.G)
        print('D:')
        utils.print_network_num_parameters(self.D)
        print('R:')
        utils.print_network_num_parameters(self.R)
        
        
    
                                 
    def train(self, flag_save_per_epoch = True, flag_visualize_per_epoch = True, R_delay=10):
        
        print('With supervised signals!' if self.supervised else 'No supervised signals.')
        print('With label training!' if self.label_training else 'No label training.')
        print('Delay R training after %d epochs.' % R_delay)
        print('Training begins')
        
        total_step = len(self.trainloader)
        self.recordLossValues(0.1, 0.1, 0.1)
        
        G_loss, D_loss, R_loss = Variable(torch.FloatTensor([0.1])), Variable(torch.FloatTensor([0.1])), Variable(torch.FloatTensor([0.1]))
        
        for epoch in tnrange(self.num_epoch, desc='epoch loop'):

            self.G.train()
            self.R.train()
            self.D.train()
        
            if epoch == R_delay:
                print('Come to the beginning of R training. Reintialize the optimizers.')
                self.initializeOptimizaer_GDR() 
        
            # for each training step
            for step, (batch_x, batch_y) in tqdm_notebook(enumerate(self.trainloader, 0), desc='step loop', leave=False):  
                
                # preprocess
                batch_x, batch_y = self.preprocess_xy(batch_x, batch_y)
                
                if self.WGAN:
                    D_loss, G_loss = self.WGANtraining(batch_x, batch_y)
                else:
                    D_loss, G_loss = self.GANtraining(batch_x, batch_y)
                
                if epoch >= R_delay:
                    R_loss = self.GRtraining(num_train_G = self.WGAN_num_critics)
                # end if epoch >= R_delay
                
                if ((step + 1) % 100) == 1:
                    D_loss_ = D_loss.data[0]
                    G_loss_ = G_loss.data[0]
                    R_loss_ = R_loss.data[0]
                    self.recordLossValues(D_loss_, G_loss_, R_loss_)
                                        
                    epoch_step = epoch * total_step + step
                    self.tbx_writer.add_scalars('D_G_loss', { 'D_loss':D_loss_, 'G_loss':G_loss_ }, epoch_step)
                    self.tbx_writer.add_scalar('R_loss', R_loss_, epoch_step)

                if ((step + 1) % 500) == 1:
                    print("Epoch: [%d] Step: [%d/%d] D_loss: %.8f, G_loss: %.8f, R_loss: %.8f" %
                          ((epoch + 1), (step + 1), total_step, D_loss_, G_loss_, R_loss_ ))
                    
                    if epoch >= R_delay and self.num_unblock < (self.code_dim + self.noise_dim):
                        self.num_unblock += 1
                    
            # end step loop
                
            if flag_save_per_epoch:
                self.save()
            
            if flag_visualize_per_epoch:
                self.visualize_results(epoch+1)
                
            self.tbx_writer.add_text('Epoch', str(epoch), epoch) 
            
            self.tbx_writer.add_image('Generated images (random)', self.sample( fixed_noise_label = False )[0], epoch)
            self.tbx_writer.add_image('Generated images (fixed)' , \
                self.sample( code = self.fixed_code, fixed_noise_label = True )[0], epoch )
            
                
        # end epoch loop        
        print("Training finished!")
        self.tbx_writer.add_text('Epoch', "Training finished!", epoch+1) 
        
        
        self.tbx_writer.export_scalars_to_json(os.path.join(self.tbx_path, 'final_checkpoint'))
        print("TensorboardX saving completed!")

        
    # end train()

    
    
    def recordLossValues(self, D_loss, G_loss, R_loss):
        
        if not hasattr(self, 'train_hist'):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['R_loss'] = []
        
        self.train_hist['D_loss'].append(D_loss)
        self.train_hist['G_loss'].append(G_loss)
        self.train_hist['R_loss'].append(R_loss)


        
    def save(self):
        save_dir = self.save_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(self.R.state_dict(), os.path.join(save_dir, self.model_name + '_R.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        
        print("Models saving completed!")
        
        
        
    def load(self):
        save_dir = self.save_path

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        self.R.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_R.pkl')))
        
        print('Models loading completed!')
        
        