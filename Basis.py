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
import utils, utils_torch
import tensorboardX
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import scipy.linalg as linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn.preprocessing import normalize
from scipy.spatial import minkowski_distance
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import StandardScaler


torch.manual_seed(2333333)    # reproducible

class Basis(object):
    def __init__(self, epoch = 20, batch_size = 64, dataset = 'mnist', save_dir = 'save_dir', gpu_mode = False, noise_dim = 10, code_dim = 2, num_labels = 10, lr = 1e-5, weight_decay = 0, supervised = True, label_training = True, tmp = True, **kwargs):
        
        # configuration
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.dataset = dataset

        # model parameters
        self.noise_dim, self.code_dim, self.num_labels = noise_dim, code_dim, num_labels
        
        # training parameter
        self.num_epoch, self.batch_size, self.gpu_mode = epoch, batch_size, gpu_mode
        self.lr, self.weight_decay = lr, weight_decay
        
        # data
        self.trainloader, self.testloader = utils.load_dataset(self.dataset, batch_size = self.batch_size )
        

        # model
        self.supervised = supervised
        self.label_training = label_training


        # other initialisation
        
        """ fixed noise to get visualisation """
        self.num_visual_samples = max(36, self.num_labels)
        
        self.num_unblock = self.noise_dim + self.code_dim
        self.fixed_noise = self.get_noise(self.num_visual_samples)
        self.fixed_code = self.get_code(self.num_visual_samples)


    def to_cuda(self):
        """ Transfer to gpu """
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

    
    def initializeTensorboardXWriter(self, desc=''):
    
        self.tbx_path = os.path.join(self.save_path, 'tbx ' + str( datetime.datetime.now() ).replace(':','-' ) + ' ' + desc )
        self.tbx_writer = tensorboardX.SummaryWriter(log_dir=self.tbx_path, comment='Start from ' + str(datetime.datetime.now()))
        # print('TensorboardX writer initialized with saved path: ' + self.tbx_path)
        
        print('')
        print('Use to following command to run the tensorboardX.')
        print('tensorboard --logdir \''+ os.path.abspath(self.tbx_path) + '\'')
        print('')
        
        self.tbx_writer.add_text('summary',\
                                 'model:{}, \n save: {}, \n dataset: {}, \n save_path:{}, \n noise:{}, \n code:{}, \n  num_labels:{}'.format(self.model_name, self.save_dir, self.dataset, self.save_path, self.noise_dim, self.code_dim, self.num_labels ) )
         

    def printNetArch(self):
        print('---------- Networks architecture -------------')
        print('G:')
        utils.print_network(self.G)
        print('D:')
        utils.print_network(self.D)
        print('-----------------------------------------------')

    def print_net_num_parameters(self):
        print('G:')
        utils.print_network_num_parameters(self.G)
        print('D:')
        utils.print_network_num_parameters(self.D)
        
        

    
    def get_noise(self, num_ = 1):
        out_noise = utils.gen_noise_Gaussian(num_, self.noise_dim)
        num_noise_unblock = max(self.num_unblock - self.code_dim, 0)
        if num_noise_unblock < self.noise_dim:
            out_noise[:, num_noise_unblock:] = 0
        return out_noise
    
    
    def get_code(self, num_ = 1):
        out_code = utils.gen_noise_Gaussian(num_, self.code_dim)
        if self.num_unblock < self.code_dim:
            out_code[:, self.num_unblock:] = 0
        return out_code
    
    
    def get_code_pair(self, num_, diff_dim=[0], interval = 0):
        first = self.get_code(num_)
        second = torch.zeros(first.size())
        second.copy_(first)
        pivot = int(num_) // 2

        lower, upper = -2, 2
        if interval == 0:
            interval = random.uniform(1,2)
        
        for i1 in diff_dim:    
            first[0:pivot, i1]  = utils.gen_noise_Gaussian(pivot, 1)
            second[0:pivot, i1] = first[0:pivot, i1] + interval

            first[pivot:, i1]  = utils.gen_noise_Gaussian(num_ - pivot, 1)
            second[pivot:, i1] = first[pivot:, i1] - interval

        return first, second
        

    def get_labels(self, num_, random = True):
        if not random:
            ll = torch.arange(0, self.num_labels).repeat(1, num_ // self.num_labels + 1)[0,:num_].long()
        else:
            ll = utils.gen_random_labels( num_, self.num_labels )
        
        out_labels = utils.long_tensor_to_onehot(ll, self.num_labels).float()
        return ll, out_labels
        

    def preprocess_xy(self, batch_x, batch_y):

        # preprocess
        batch_x = batch_x.float()
        batch_y = batch_y.long().squeeze()
        if self.num_labels == 1:
            batch_y.zero_()
        return batch_x, batch_y


    def recordLossValues_dict(self, **kwargs):

        if not hasattr(self, 'train_hist'):
            self.train_hist = {}
            
        for key in kwargs.keys():
            if not hasattr(self.train_hist, key):
                self.train_hist[key] = []
        
        for key in kwargs.keys():
            self.train_hist[key].append(kwargs[key])


    def getDataThisBatch(self, batch_x, batch_y):
        
        this_batch_size = batch_x.size()[0]
        var_y_real = utils_torch.to_var(batch_y , gpu_mode = self.gpu_mode)
        var_y_real_onehot = utils_torch.to_var(utils.long_tensor_to_onehot(batch_y, self.num_labels).float(), gpu_mode = self.gpu_mode)
        var_x_real = utils_torch.to_var(batch_x, gpu_mode = self.gpu_mode)
        var_ones_batch = utils_torch.to_var(torch.ones(this_batch_size,1), gpu_mode = self.gpu_mode)
        var_zeros_batch = utils_torch.to_var(torch.zeros(this_batch_size,1), gpu_mode = self.gpu_mode)
        
        return var_x_real, var_y_real, var_y_real_onehot, var_ones_batch, var_zeros_batch
    
    
    def getNoiseThisBatch(self, this_batch_size):
        var_z_noise = utils_torch.to_var( self.get_noise(this_batch_size), gpu_mode = self.gpu_mode )
        var_z_code = utils_torch.to_var( self.get_code(this_batch_size), gpu_mode = self.gpu_mode )
        var_z_labels = utils_torch.to_var(self.get_labels(this_batch_size)[1], gpu_mode = self.gpu_mode )
        return var_z_noise, var_z_code, var_z_labels



    def GANtraining(self, batch_x, batch_y):
        this_batch_size = batch_x.size()[0]
        var_z_1_noise, var_z_1_code, _ = self.getNoiseThisBatch( this_batch_size )
        var_x_real, var_y_real, var_y_real_onehot, var_ones_batch, var_zeros_batch = self.getDataThisBatch(batch_x, batch_y)
        
        # train D
        self.D_optimizer.zero_grad()
        var_D_real, var_D_real_labels = self.D(var_x_real)[0:2]
        
        D_real_loss = self.BCE_loss(var_D_real.squeeze(), var_ones_batch.squeeze())
        D_real_labels_loss = self.CE_loss( var_D_real_labels, var_y_real) if self.supervised else 0
        D_real_loss_and_label_loss = D_real_loss + D_real_labels_loss
        
        var_x_fake = self.G(var_z_1_noise, var_z_1_code, var_y_real_onehot).detach()
        var_D_fake = self.D(var_x_fake)[0]

        D_fake_loss = self.BCE_loss(var_D_fake.squeeze(), var_zeros_batch.squeeze())
        
        D_loss = D_real_loss_and_label_loss + D_fake_loss

        D_loss.backward()
        self.D_optimizer.step()

        D_loss = D_real_loss + D_fake_loss

        
        # train G

        self.G_optimizer.zero_grad()

        var_z_1_noise, var_z_1_code, _ = self.getNoiseThisBatch( this_batch_size )
        var_x_fake = self.G(var_z_1_noise, var_z_1_code, var_y_real_onehot)
        var_D_fake, var_D_fake_labels = self.D(var_x_fake)[0:2]

        G_loss = self.BCE_loss(var_D_fake.squeeze(), var_ones_batch.squeeze())
        G_loss.backward(retain_graph = True)
        self.G_optimizer.step()

        if self.label_training:
                # train label loss
                self.G_optimizer.zero_grad()
                self.D_optimizer.zero_grad()
                label_loss = self.CE_loss(var_D_fake_labels, var_y_real)
                label_loss.backward()
                self.G_optimizer.step()
                self.D_optimizer.step()

        return D_loss, G_loss

    
    def WGANtraining(self, batch_x, batch_y, wgan_clamp = 0.01):

        # data in this batch
        var_x_real, var_y_real, var_y_real_onehot, var_ones_batch, var_zeros_batch = self.getDataThisBatch(batch_x, batch_y)
        this_batch_size = batch_x.size()[0]
        
        # counts
        if not hasattr(self, 'D_iter') and not hasattr(self, 'G_iter'):
            self.D_iter, self.G_iter = 0, 0
        
        # adjust the num_critics
        num_critics_ = self.WGAN_num_critics
        if self.G_iter < 25 or self.G_iter % 200 == 0:
            num_critics_ = 100
        
        # train D ==========================================
        
        # fake samples
        var_z_noise, var_z_code, _ = self.getNoiseThisBatch(this_batch_size)
        var_x_fake = self.G(var_z_noise, var_z_code, var_y_real_onehot)
        var_x_fake.detach_()
        
        if not self.WGAN_GP:
            # clipping D
            for p in self.D.parameters():
                p.data.clamp_(-wgan_clamp, wgan_clamp)
            gradient_penalty = 0
        else:
            # gradient penalty, need the instance normalization
            lambda_ = 0.25
            alpha = random.uniform(0, 1)
            var_x_hat = Variable(alpha * var_x_real.data + (1 - alpha) * var_x_fake.data, requires_grad=True)
            var_pred_hat = self.D(var_x_hat)[0]
            gradients = grad(outputs=var_pred_hat, inputs=var_x_hat, grad_outputs=var_ones_batch,
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        
        # real loss
        var_D_real, var_D_real_labels = self.D(var_x_real)[0:2]
        D_real_loss = - torch.mean(var_D_real)
        D_real_labels_loss = self.CE_loss(var_D_real_labels, var_y_real) if self.supervised else 0

        # fake loss
        var_D_fake = self.D(var_x_fake)[0]
        D_fake_loss = torch.mean(var_D_fake)
        
        # total loss
        D_loss = D_real_loss + D_fake_loss + D_real_labels_loss + gradient_penalty
        
        # train
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()
        
        # count this run
        self.D_iter += 1
        
        # the real D_loss
        D_loss = (- D_real_loss) - D_fake_loss

        # train G ==========================================
        
        # default loss if not trained
        G_loss = - D_fake_loss
        
        if self.D_iter % num_critics_ == 0:

            # fake samples
            var_z_noise, var_z_code, _ = self.getNoiseThisBatch(this_batch_size)
            var_x_fake = self.G(var_z_noise, var_z_code, var_y_real_onehot)
            
            # G_loss
            var_D_fake, var_D_fake_labels = self.D(var_x_fake)[0:2]
            G_loss = - torch.mean(var_D_fake)
            
            # update
            self.G_optimizer.zero_grad()
            G_loss.backward(retain_graph = self.label_training)
            self.G_optimizer.step()
            
            
            # count this run
            self.G_iter += 1

            # in train G, train labels            

            if self.label_training:
                    # train label loss
                    self.G_optimizer.zero_grad()
                    self.D_optimizer.zero_grad()
                    label_loss = self.CE_loss(var_D_fake_labels, var_y_real)
                    label_loss.backward()
                    self.G_optimizer.step()
                    self.D_optimizer.step()
            

        return D_loss, G_loss

    
    
    def GRtraining(self, num_train_G = 1):
        
        this_batch_size = 8 # self.batch_size // 2
        R_loss = Variable(torch.zeros(1))
        
        # record the number of iter
        if not hasattr(self, 'R_iter'):
            self.R_iter = 0
            self.R_training_up_to_code = 1
            # self.R_interval = 2
        self.R_iter += 1
        
        self.R_training_up_to_code = min(self.num_unblock, self.code_dim)
        
        # train R and G
        ''' train to find which code dim is different '''

        diff_dims_batch_list = []
        first_code_batch_list = []
        second_code_batch_list = []
        
        for diff_dim in range(self.R_training_up_to_code):
            diff_dims = torch.LongTensor([ diff_dim ])
            diff_dims_batch_list.append(diff_dims.repeat(this_batch_size))
            
            first_code, second_code = self.get_code_pair(this_batch_size, diff_dim = diff_dims)
            
            first_code_batch_list.append(first_code)
            second_code_batch_list.append(second_code)
        
        # the total diff codes        
        first_code, second_code = torch.cat(first_code_batch_list, 0), torch.cat(second_code_batch_list, 0)
        
        var_diff_code_1 = utils_torch.to_var(first_code, gpu_mode=self.gpu_mode)
        var_diff_code_2 = utils_torch.to_var(second_code, gpu_mode=self.gpu_mode)
        
        # the diff dim
        diff_dims_batch = torch.cat(diff_dims_batch_list, 0)
        var_diff_dims_batch = utils_torch.to_var(diff_dims_batch, gpu_mode=self.gpu_mode)

        # fixed noise and labels
        var_z_noise, _, var_z_labels = self.getNoiseThisBatch(this_batch_size * self.R_training_up_to_code)
        
        var_z_noise_1 = utils_torch.to_var(var_z_noise.data, gpu_mode = self.gpu_mode)
        var_z_noise_2 = utils_torch.to_var(var_z_noise.data, gpu_mode = self.gpu_mode)
        var_z_noise_1.detach_()
        var_z_noise_2.detach_()
        

        # generated sample pairs
        var_x_diff_1 = self.G(var_z_noise_1, var_diff_code_1, var_z_labels)
        var_x_diff_2 = self.G(var_z_noise_2, var_diff_code_2, var_z_labels)

        # output representation
        ID12 = self.R(var_x_diff_1, var_x_diff_2)

        # try to find which dim is different
        R_loss = self.CE_loss(ID12, var_diff_dims_batch ) 

        # whether we should train G in this step        
        train_G = (self.R_iter % num_train_G) == 0
        
        # if not, it is a waste to backward the gradients to G
        if not train_G:
            var_x_diff_1.detach_()
            var_x_diff_2.detach_()
        
        # get the gradient and update
        self.G_optimizer.zero_grad()
        self.R_optimizer.zero_grad()
        
        R_loss.backward()
        self.R_optimizer.step()

        if train_G:
            self.G_optimizer.step()
        
        return R_loss



        
    def save(self):
        save_dir = self.save_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        
        print("Models saving completed!")
        
        
        
    def load(self):
        save_dir = self.save_path

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        
        print('Models loading completed!')
        


    def visualize_results(self, epoch):
        """ visualize the sampled results and save them"""

        save_path = os.path.join(self.save_path, 'visualization')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        samples_grid = self.sample(code = self.fixed_code, fixed_noise_label = True )[0]

        plt.imshow( utils_torch.to_np(samples_grid).transpose(1, 2, 0) )
        plt.show()

        torchvision.utils.save_image(samples_grid, filename = os.path.join(save_path, 'epoch%03d' % epoch + '.png') )

        print('Sampled images saved.')

    def generate_animation(self, epoch):
        
        for i1 in range(self.code_dim):
            utils.generate_animation(self.save_path+'/visualization/code%02d' % i1, epoch)
        
        print("Animations saved")
    

    def generate_code_animation(self, epoch = 0):
        """ generate the animation of the generated pictures when code varies. """
        
        save_path = os.path.join(self.save_path, 'visualization', 'code_epoch%03d' % epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                
        print("Start generating code animations.")

        # the num of code to change
        num_range = 61
        code_variant = torch.linspace(-3, 3, num_range)

        # generate animation for every code
        for i1 in tnrange(self.code_dim):
            images = []
            # each sub picture
            for i2 in range(num_range):
                
                this_code = torch.zeros( self.num_visual_samples, self.code_dim )
                this_code.copy_(self.fixed_code)
                this_code[:,i1] = code_variant[i2]

                samples_grid = self.sample(code = this_code, fixed_noise_label = True)[0]

                images.append( utils_torch.to_np(samples_grid*255).astype(np.uint8).transpose(1, 2, 0)  )

            imageio.mimsave( os.path.join(save_path, 'generate_animation_code%03d.gif' % i1), images, fps = 24)
        
        print("Code animations generated and saved.")

        
    
    def generate_noise_animation(self, epoch = 0):
        """ generate the animation of the generated pictures when noise varies. """
        
        save_path = os.path.join(self.save_path, 'visualization', 'noise_epoch%03d' % epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                
        print("Start generating noise animations.")

        # the num of noise to change
        num_range = 61
        noise_variant = torch.linspace(-3, 3, num_range)

        # generate animation for every noise
        for i1 in tnrange(self.noise_dim):
            images = []
            # each sub picture
            for i2 in range(num_range):
                
                this_noise = torch.zeros( self.num_visual_samples, self.noise_dim )
                this_noise.copy_(self.fixed_noise)
                this_noise[:,i1] = noise_variant[i2]

                in_noise = utils_torch.to_var(this_noise, gpu_mode = self.gpu_mode,  volatile = True)
                in_code = utils_torch.to_var(torch.zeros( self.num_visual_samples, self.code_dim ), gpu_mode = self.gpu_mode,  volatile = True)
                # the fixed labels
                in_labels = utils_torch.to_var( self.get_labels(self.num_visual_samples, random = False)[1], gpu_mode = self.gpu_mode,  volatile = True)

                self.G.eval()
                samples = self.G(in_noise, in_code, in_labels)

                r = int( np.sqrt( self.num_visual_samples ) )

                samples_grid = torchvision.utils.make_grid(samples.cpu().data, nrow = r, normalize = True, padding = 4 ).float()

                images.append( utils_torch.to_np(samples_grid*255).astype(np.uint8).transpose(1, 2, 0)  )

            imageio.mimsave( os.path.join(save_path, 'generate_animation_noise%03d.gif' % i1), images, fps = 24)
        
        print("Noise animations generated and saved.")
        
        
    

    def sample(self, code = torch.zeros(1), fixed_noise_label = True, zero_noise = True):
        """ sample num_visual_samples images from current G model. """

        # assume everything random
        in_noise, in_code, in_labels = self.getNoiseThisBatch( self.num_visual_samples )
        
        # if given code, then assign
        if not code.equal( torch.zeros(1) ):
            in_code = utils_torch.to_var(code, gpu_mode = self.gpu_mode,  volatile = True)

        # if wanna fix noise and label, then zeros
        if fixed_noise_label:
            # the fixed noise
            if zero_noise:
                in_noise = utils_torch.to_var(torch.zeros(in_noise.size()), gpu_mode = self.gpu_mode,  volatile = True)
            else:
                in_noise = utils_torch.to_var(self.fixed_noise, gpu_mode = self.gpu_mode,  volatile = True)
            
            # the fixed labels
            in_labels = utils_torch.to_var( self.get_labels(self.num_visual_samples, random = False)[1], gpu_mode = self.gpu_mode,  volatile = True)
                        
        self.G.eval()
        samples = self.G(in_noise, in_code, in_labels)

        r = int( np.sqrt( self.num_visual_samples ) )

        samples_grid = torchvision.utils.make_grid(samples.cpu().data, nrow = r, normalize = True, padding = 4 ).float()

        return samples_grid, samples

    def sample_with_code(self, in_code ):
        num_ = len( in_code )
        self.G.eval()
        var_z_noise, _, var_z_labels = self.getNoiseThisBatch(num_)
        var_z_noise  = utils_torch.to_var(var_z_noise.data, volatile = True, gpu_mode = self.gpu_mode)
        var_z_labels = utils_torch.to_var(var_z_labels.data, volatile = True, gpu_mode = self.gpu_mode)
        var_z_code = utils_torch.to_var(in_code, volatile = True, gpu_mode = self.gpu_mode)
        out = self.G( var_z_noise, var_z_code, var_z_labels )
        return out.cpu().data
        
    
    def getDisentanglementMetric(self, alpha = 0.5, real_sample_fitting_method = 'LinearRegression', subspace_fitting_method = 'OMP'):
        print('Begin the calculation of the subspace score.')
        
        self.G.eval()
        
        num_tries = 5
        # code variantion range
        num_range = 5
        code_variant = torch.linspace(-2, 2, num_range)
        # the number of samples per batch and the result
        num_sample_per_batch = 10
        
        class Reconstructor():
            def __init__(self, reconstruct_method):
                super().__init__()
                self.reconstruct_method = reconstruct_method

                method = {'LinearRegression': LinearRegression(fit_intercept=False, n_jobs = -1),
                          'Ridge': Ridge(alpha=1e-4, fit_intercept=False, tol=1e-2,), 
                          'Lasso': Lasso(alpha=1e-5, fit_intercept=False, warm_start = False, tol=1e-3,),
                          'ElasticNet': ElasticNet(alpha=1e-2, l1_ratio = 0.5, fit_intercept=False, tol=1e-3,),
                          'OMP': OMP(n_nonzero_coefs = num_range * num_sample_per_batch, fit_intercept=False ),
                         }
                self.clf = method[reconstruct_method]

                print('Reconstructor initialized with the method as %s' % reconstruct_method)

            def fit(self,X,Y):
                X, Y = normalize(X, axis=1), normalize(Y, axis=1) # unit length
                Xt, Yt = np.transpose(X), np.transpose(Y)
                # print('Reconstructor fit() called. Begin to fit from %s to %s.' % (str(X.shape), str(Y.shape)) )
                self.clf.fit(Xt, Yt)
                Y_hat = self.clf.coef_ @ X #+ self.clf.intercept_.reshape(1,-1)
                return np.mean(minkowski_distance(Y, Y_hat)), self.clf.coef_, Y_hat


            def fit_self(self,X):
                # print('Reconstructor fit_self() called. Begin to fit %s' % str(X.shape) )
                X = normalize(X, axis=1) # unit length
                num_sample = len(X)
                idx = np.arange(num_sample)
                result_matrix = np.zeros([num_sample, num_sample])
                for i1 in tnrange(num_sample, leave=False):
                    this_idx = np.delete(idx, i1 ).tolist()
                    coef = self.fit(X[this_idx, :], X[i1, :].reshape(1,-1))[1]
                    result_matrix[i1, this_idx] = coef
                return result_matrix
        

        final_result_batch = []
        for i0 in tnrange(num_tries, desc = 'total rounds'):

            # normalizer
            scaler = StandardScaler(with_mean=True, with_std=False)

            # reconstructor 
            print('Real samples fitting method')
            r_fit_real = Reconstructor(real_sample_fitting_method)
            print('Generated samples fitting method')
            r_fit_generated = Reconstructor(subspace_fitting_method)

            # get some of the real samples
            trainloader_iter = iter(self.trainloader)
            part_of_real_samples = torch.cat( [ next(trainloader_iter)[0] for _ in range(200) ] )
            part_of_real_samples_np = part_of_real_samples.view(part_of_real_samples.size()[0], -1).numpy()
            part_of_real_samples_np = scaler.fit_transform(part_of_real_samples_np)

            total_samples_batch = []
            total_labels_batch = []
            # generate sequences for every code
            for i1 in tnrange(self.code_dim):
                in_noise, in_code, in_labels = self.getNoiseThisBatch( num_sample_per_batch )
                this_code = self.get_code( num_sample_per_batch )#.zero_()
                # this_code = torch.zeros(this_code.size())
                # each code varies
                for i2 in range(num_range):
                    this_code[:,i1] = code_variant[i2]
                    in_code = utils_torch.to_var(this_code, gpu_mode = self.gpu_mode, volatile = True)
                    samples = self.G(in_noise, in_code, in_labels).data                    
                    total_samples_batch.append( samples )
                    total_labels_batch.append( torch.ones(num_sample_per_batch) * i1 )

            # all the generated sequence
            total_samples = torch.cat(total_samples_batch)
            total_labels = torch.cat(total_labels_batch)

            
            
            # numpy format
            total_samples_np = utils_torch.to_np( total_samples.view(total_samples.size()[0], -1) ) 
            total_samples_np = scaler.transform(total_samples_np)
            total_labels_np = utils_torch.to_np( total_labels )

            # print('Begin to fit real samples.')
            # the reconstruction accuracy of the real samples from the generated samples
            reconstructed_accuracy = r_fit_real.fit(total_samples_np, part_of_real_samples_np)[0]
            
            # print('Begin to fit generated samples.')
            # fit the total_samples_np with itself
            coefficient_matrix = r_fit_generated.fit_self(total_samples_np)

            # symmetrization
            coefficient_matrix_abs = np.abs(coefficient_matrix)
            coefficient_matrix_sym = coefficient_matrix_abs + np.transpose(coefficient_matrix_abs)
            
            # # show the ``covariance'' matrix
            # plt.imshow( coefficient_matrix_sym / np.max(coefficient_matrix_sym) )
            # plt.show()
            
            self.tbx_writer.add_image('coefficient_matrix', torch.from_numpy(coefficient_matrix_sym).view(1,1,total_samples.size()[0],total_samples.size()[0]), i0)
            
            # subspace clustering 
            label_hat = spectral_clustering(coefficient_matrix_sym, n_clusters = self.code_dim)

            NMI = metrics.normalized_mutual_info_score(label_hat, total_labels_np)
            
            final_result_this = (1 - reconstructed_accuracy) * alpha + NMI * (1 - alpha)
            
            to_print = 'ROUND {}:\n distance to projection:{}\n NMI:{}\n final result:{}\n'.format(i0, reconstructed_accuracy, NMI, final_result_this)
            # print( to_print )
            self.tbx_writer.add_text('disentanglement metric', to_print, i0)

            final_result_batch.append(final_result_this)

        to_print = 'final subspace score value: {}+-{}'.format( np.mean(final_result_batch), np.std(final_result_batch) ) 
        print( to_print)
        self.tbx_writer.add_text('disentanglement metric', to_print, num_tries)

        return np.mean(final_result_batch)