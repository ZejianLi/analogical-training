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

class ANA_VAE(Basis):
    def __init__(self, beta = 1,  **kwargs):

        super(ANA_VAE, self).__init__(**kwargs)
        
        print('INFO: using VAE with beta as {}'.format(beta))
        
        # configuration
        self.model_name = 'ANA_VAE'
        self.save_path = os.path.join(self.save_dir, self.dataset, self.model_name)

        # data
        self.trainloader, self.testloader = utils.load_dataset(self.dataset, batch_size = self.batch_size, for_tanh = False )
        
        # model
        self.G, self.E, self.R = Models_interface.get_GER_model(dataset = self.dataset,
            code_dim = self.code_dim, 
            noise_dim = self.noise_dim, 
            label_dim = self.num_labels, 
            supervised = self.supervised,
            Tanh = False)
        
        # parameters
        self.beta = beta
        self.num_unblock = self.code_dim
        
        # optimizer
        self.initializeOptimizaer_GER()
        
        
        self.to_cuda()
        if self.gpu_mode:
            self.R.cuda()
        
    def to_cuda(self):
        """ Transfer to gpu """
        if self.gpu_mode:
            self.G.cuda()
            self.E.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

                
    def get_noise(self, num_ = 1):
        return torch.zeros(num_, self.noise_dim)

    
    def printNetArch(self):
        print('---------- Networks architecture -------------')
        print('G:')
        utils.print_network(self.G)
        print('E:')
        utils.print_network(self.E)
        print('R:')
        utils.print_network(self.R)
        print('-----------------------------------------------')

    def print_net_num_parameters(self):
        print('G:')
        utils.print_network_num_parameters(self.G)
        print('E:')
        utils.print_network_num_parameters(self.E)
        print('R:')
        utils.print_network_num_parameters(self.R)
        

    def initializeOptimizaer_GER(self):
        
        beta1, beta2 = 0.5, 0.99
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
        self.R_optimizer = optim.Adam(self.R.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = self.weight_decay)
        print('Use ADAM optimizers.')
        

    def train(self, flag_save_per_epoch = True, flag_visualize_per_epoch = True, R_delay=10):
        
        print('With supervised signals!' if self.supervised else 'No supervised signals.')
        print('Delay R training after %d epochs.' % R_delay)
        print('Training begins')
        
        total_step = len(self.trainloader)
        self.recordLossValues(0.1, 0.1, 0.1)
        
        G_loss, D_loss, R_loss = Variable(torch.FloatTensor([0.1])), Variable(torch.FloatTensor([0.1])), Variable(torch.FloatTensor([0.1]))
        
        for epoch in tnrange(self.num_epoch, desc='epoch loop'):

            self.G.train()
            self.R.train()
            self.E.train()
        
            if epoch == R_delay:
                print('Come to the beginning of R training. Reintialize the optimizers.')
                self.initializeOptimizaer_GER() 
        
            # for each training step
            for step, (batch_x, batch_y) in tqdm_notebook(enumerate(self.trainloader, 0), desc='step loop', leave=False):  
                
                # preprocess
                batch_x, batch_y = self.preprocess_xy(batch_x, batch_y)
                
                # vae training
                reconstruction_loss, disentangled_loss, D_real_labels_loss, var_out = self.VAE_training(batch_x, batch_y)
               
                # R training
                if epoch >= R_delay:
                    R_loss = self.GRtraining(num_train_G = 1)
                # end if epoch >= R_delay


                if ((step + 1) % 100) == 1:
                    reconstruction_loss_ = reconstruction_loss.data[0]
                    disentangled_loss_ = disentangled_loss.data[0]
                    D_real_labels_loss_ = D_real_labels_loss.data[0] if self.supervised else 0
                    R_loss_ = R_loss.data[0]
                    
                    self.recordLossValues(reconstruction_loss_, disentangled_loss_, R_loss_)
                                        
                    epoch_step = epoch * total_step + step
                    self.tbx_writer.add_scalars('reconstruction_loss_disentangled_loss', \
                        { 'reconstruction_loss':reconstruction_loss_, 'disentangled_loss':disentangled_loss_ }, epoch_step)
                    self.tbx_writer.add_scalar('total_loss', reconstruction_loss_ + disentangled_loss_, epoch_step)
                    self.tbx_writer.add_scalar('D_real_labels_loss', D_real_labels_loss_, epoch_step)
                    self.tbx_writer.add_scalar('R_loss', R_loss_, epoch_step)
                    
    
                if ((step + 1) % 500) == 1:
            
                    print("Epoch: [%d] Step: [%d/%d] reconstruction_loss: %.8f, disentangled_loss: %.8f, R_loss: %.8f" %
                          ((epoch + 1), (step + 1), total_step, reconstruction_loss_, disentangled_loss_, R_loss_ ))
                
                if ((step + 1) % 1000) == 1:
                    samples_grid = torchvision.utils.make_grid( torch.cat([var_out.cpu().data, batch_x], 0), nrow = 8, normalize = True, padding = 4 ).float()
                    self.tbx_writer.add_image('Reconstructed images', samples_grid, epoch_step)
                    

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



    def reparametrize(self, mu, log_var):
        """" z = mean + eps * sigma where eps is sampled from N(0, 1). """
        eps = utils_torch.to_var( torch.randn( mu.size(0), mu.size(1) ), gpu_mode=self.gpu_mode )
        z = mu + eps * torch.exp( log_var / 2 )    # 2 for convert var to std
        return z

    def kl_divergence(self, mu, log_var):
        return torch.sum( 0.5 * (mu**2 + torch.exp(log_var) - log_var - 1) )
    
    def VAE_training(self, batch_x, batch_y):
        # to variable
        this_batch_size = batch_x.size()[0]
        var_x_real, var_y_real, var_y_real_onehot, _, _ = self.getDataThisBatch(batch_x, batch_y)
        var_z_noise, var_z_code, _ = self.getNoiseThisBatch(this_batch_size)
        
        # hidden representation
        var_code, var_noise, var_D_real_labels  = self.E(var_x_real)

        # supervised training
        D_real_labels_loss = F.cross_entropy(var_D_real_labels, var_y_real, size_average=False) if self.supervised else 0
        
        # divided into mu and log variance
        mu_code,  log_var_code  = torch.chunk(var_code, 2, dim=1)  # mean and log variance.

        # number of noise unblock and get the intermediate loss
        disentangled_loss = self.kl_divergence(mu_code, log_var_code) 

        # reparameterization
        var_z_code = self.reparametrize(mu_code, log_var_code)
        
        # generated samples
        var_out = self.G(var_z_code, var_z_noise, var_y_real_onehot)
        
        reconstruction_loss = F.binary_cross_entropy(var_out, var_x_real, size_average=False)
        
        # final loss
        total_loss = reconstruction_loss + self.beta * disentangled_loss + D_real_labels_loss

        # update
        self.G_optimizer.zero_grad()
        self.E_optimizer.zero_grad()
        total_loss.backward()
        self.G_optimizer.step()
        self.E_optimizer.step()

        return reconstruction_loss, disentangled_loss, D_real_labels_loss, var_out


    def recordLossValues(self, reconstruction_loss, disentangled_loss, R_loss):
        
        if not hasattr(self, 'train_hist'):
            self.train_hist = {}
            self.train_hist['reconstruction_loss'] = []
            self.train_hist['disentangled_loss'] = []
            self.train_hist['R_loss'] = []
        
        self.train_hist['reconstruction_loss'].append(reconstruction_loss)
        self.train_hist['disentangled_loss'].append(disentangled_loss)
        self.train_hist['R_loss'].append(R_loss)


    def save(self):
        save_dir = self.save_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_E.pkl'))
        torch.save(self.R.state_dict(), os.path.join(save_dir, self.model_name + '_R.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        
        print("Models saving completed!")
        
        
    def load(self):
        save_dir = self.save_path

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_E.pkl')))
        self.R.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_R.pkl')))
        
        print('Models loading completed!')

        
        