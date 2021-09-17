
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)    
import time

import sys
sys.path.append('../utils')
import os

# from utils.pytorch_fid.fid_score import FID
from datetime import datetime
import json
date = str(datetime.now()).replace(' ', '_')[:-7].replace(':', '-')
import torchvision.utils as vutils

from utils.losses import Conditional_Contrastive_loss
from utils.sample import sample_latents, sample_1hot, make_mask, target_class_sampler
from utils.misc import toggle_grad

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class Pad():
  def __init__(self):
    return
      
  def __call__(self, tensor):
    self.tensor = tensor
    channels = tensor.shape[1] # num channels in batch
    
    if channels == 3: 
      npad  = ((0,0), (1, 0), (0, 0), (0, 0))
      # self.tensor = np.pad(self.tensor, pad_width=npad, mode='constant', constant_values=0)
    elif channels == 1:
      self.tensor = torch.cat((self.tensor.data, self.tensor.data, self.tensor.data), dim=1)
      npad  = ((0,0), (1, 0), (0, 0), (0, 0))

    self.tensor = np.pad(self.tensor, pad_width=npad, mode='constant', constant_values=0)


    return torch.Tensor(self.tensor)


class Trainer():
    def __init__(self,switch ,z_size, prior, num_classes, pos_collected_numerator, freeze_layers, conditional_strategy, batch_size,
                 generator, discriminator, gen_optimizer, dis_optimizer,
                 loss='hinge', gp_weight=10, critic_iterations=2, print_every=50,
                 use_cuda=True,
                 gpu_num=1,
                 save_FID=False,
                 FIDPaths = ['generated_images','real_images'],
                 checkpoint_folder='./ch/',
                 FIDevery = 500,
                 FIDImages = 100,
                 plot_images=False,
                 save_images=False,
                 saveModelsPerEpoch=True,
                 normalize=True):
        self.switch=switch
        self.batch_size = batch_size
        self.conditional_strategy = conditional_strategy
        self.freeze_layers = freeze_layers
        self.z_size = z_size
        self.prior = prior
        self.num_classes = num_classes
        self.pos_collected_numerator = pos_collected_numerator
        self.G = generator
        #self.G_opt = gen_optimizer
        self.D = discriminator
        if self.switch =='startfromscratch':
          self.G_opt = gen_optimizer
          self.D_opt = dis_optimizer
          self.losses = {'LossG': [], 'LossD': [], 'GP': []}
        if self.switch =='startfromcheck':
          print("sexy")
          checkpointD = torch.load('ch/MidQuaternionDiscriminator/_epoch1_GSN-False_DSN-True_2021-09-17_17-52-03.pt')
          checkpointG = torch.load('ch/MidQuaternionGenerator/_epoch1_GSN-False_DSN-True_2021-09-17_17-52-03.pt')
          self.G.load_state_dict(checkpointG['model_state_dict'])
          print(self.G)
          self.D.load_state_dict(checkpointD['model_state_dict'])
          print(self.D)
          self.epoch=checkpointD['epoch']
          print(self.epoch)
          #gen_optimizer.to(device)
          gen_optimizer.load_state_dict(checkpointG['optimizer_state_dict'])
          self.G_opt = gen_optimizer
          device = torch.device("cuda")
          optimizer_to(self.G_opt,device)
          print(self.G_opt)
          dis_optimizer.load_state_dict(checkpointD['optimizer_state_dict'])
          self.D_opt = dis_optimizer 
          optimizer_to(self.D_opt,device)
          print(self.D_opt)
          self.losses = {'LossG': [checkpointG['lossG']], 'LossD': [checkpointD['loss']], 'GP': [checkpointG['lossGP']]}
          print(self.losses['LossG'])
          print(self.losses['LossD'])
          print(self.losses['GP'])
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

#         if self.use_cuda:
        device = torch.device('cuda:%i' %gpu_num if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.device = device
        self.G.to(device)
        self.D.to(device)
        print(device)
        
        self.selected_loss = loss
        if self.selected_loss == 'classic':
            self.BCE_loss = nn.BCELoss()

        
        self.save_FID = save_FID
        # self.FID = FID()
        self.FIDPaths = FIDPaths
        self.FIDImages = FIDImages
        self.saveModelsPerEpoch = saveModelsPerEpoch
        self.save_images = save_images
        self.plot_images = plot_images # plot images during training
        self.normalize = normalize
        
        self.tracked_info = {'Epochs': 0, 'Iterations': 0, 'LossG': [], 'LossD': [], 'GP': [], 'FID': [], 'EpochFID': [], 'RotationG': [], 'RotationD': []}
        self.checkpoint_folder = checkpoint_folder
        
        
        self.QNet = ('Q' in self.G.__class__.__name__ ) # check if network is Quaternionic
        
        
        # init weight of DCGAN and QDCGAN
        # if hasattr(self.G, 'needs_init') and hasattr(self.D, 'needs_init'):
        #     if self.G.needs_init==True and self.D.needs_init==True:
        #         print('DCGAN/QDCGAN Weights init =', self.G.needs_init)
        #         if not self.QNet:
        #             self.G.apply(weights_init)
        #             # print(list(self.G.children()))
        #             self.D.apply(weights_init)
        #         else:
        #             self.G.apply(Qweights_init)
        #             # print(list(self.G.children()))
        #             self.D.apply(Qweights_init)
                   
        
        # info about Gen to put in folder's name or file name        
        self.Generator_info = self.G.__class__.__name__ + '_GSN-{}_DSN-{}'.format(
            hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm, hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm)
        
        # update generated images fid path
        self.FIDPaths[0] = str(self.FIDPaths[0]) + str(self.Generator_info)
        print('\nGenerated images saved in', self.FIDPaths[0])
       
        dir_gen = self.FIDPaths[0] #os.path.abspath(self.FIDPaths[0])
        # create folder for generated images
        if not os.path.isdir(dir_gen):
            os.makedirs(dir_gen)
            
            
        print('\nQuaternion Model = {}\nGenerator, Discriminator = {} and {}\nloss = {}\n\
               \nSpectral Normalization Generator = {}\nSpectral Normalization Discriminator = {}\n'.format(self.QNet,
              self.G.__class__.__name__, self.D.__class__.__name__, self.selected_loss,
                  hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm ==True,
                  hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm ==True))
        
        time.sleep(5)
            

    def _critic_train_iteration(self, real_images, real_labels, batch_size):
        """ Compute Discriminator Loss and Optimize """
        
        toggle_grad(self.D, on=True, freeze_layers=self.freeze_layers)
        toggle_grad(self.G, on=False, freeze_layers=-1)

        self.D.zero_grad()

        zs, fake_labels  = sample_latents(self.prior, batch_size, self.z_size, -1.0, self.num_classes, None)

        if self.use_cuda:
            zs = zs.cuda()
            fake_labels = fake_labels.cuda()
    
        fake_images = self.G(zs, fake_labels)


        if self.conditional_strategy == "ContraGAN":
            cls_proxies_real, cls_embed_real, dis_out_real = self.D(real_images, real_labels)
            cls_proxies_fake, cls_embed_fake, dis_out_fake = self.D(fake_images, fake_labels)

        # if self.selected_loss != 'classic':
        # if self.gp_weight > 0:
        #     data_up = data[0:batch_size]
        #     generated_data_up = generated_data[0:batch_size]
            
        #     # Get gradient penalty
        #     gradient_penalty = self._gradient_penalty(data_up.data, generated_data_up.data)
        #     self.losses['GP'].append(gradient_penalty.item())
           
           
        # # Create D loss and optimize
        # if self.selected_loss=='wasserstein':
        #     d_loss = torch.mean(g_fake_logits) - torch.mean(d_real_logits) 
            
        if self.selected_loss == 'hinge':
            # print('d_real_pro_logits shape', d_real_pro_logits.shape)
            d_loss = torch.mean(nn.ReLU()(1.0 - dis_out_real.view(-1))) + torch.mean(nn.ReLU()(1.0 + dis_out_fake.view(-1)))
        else:
            d_loss = 0
            print("No loss computed!")


        if self.conditional_strategy == "ContraGAN":
            t = 1.0
            margin = 0.0
            contrastive_lambda = 1.0
            real_cls_mask = make_mask(real_labels, self.num_classes, mask_negatives=self.mask_negatives, device=self.device)
            d_loss += contrastive_lambda*self.contrastive_criterion(cls_embed_real, cls_proxies_real,
                                                                            real_cls_mask, real_labels, t, margin)

        # elif self.selected_loss == 'classic':
        #     real_sigmoid, _ = self.D(data)
        #     errD = self.BCE_loss(real_sigmoid.view(-1), self.label_one)
        #     errD.backward()
        #     fake_sigmoid, _ = self.D(generated_data)
        #     errG = self.BCE_loss(fake_sigmoid.view(-1), self.label_zero)
        #     errG.backward()
        #     d_loss = errD + errG
        
        # if self.gp_weight > 0 :
        #     if self.selected_loss!= 'classic':
        #         d_loss += gradient_penalty
        #     else:
        #         d_loss == gradient_penalty
        

        if self.selected_loss != 'classic':
            d_loss.backward()#retain_graph=True)
            
        # Optimize
        self.D_opt.step()

        # Record loss
        self.losses['LossD'].append(d_loss.item())


    def _generator_train_iteration(self, batch_size):
        """ Compute Generator Loss and Optimize """
        # print('gen data size', generated_data.shape)
        toggle_grad(self.D, False, freeze_layers=-1)
        toggle_grad(self.G, True, freeze_layers=-1)

        self.G.zero_grad()
        
        zs, fake_labels = sample_latents(self.prior, batch_size, self.z_size, -1.0, self.num_classes, None)
        if self.use_cuda:
            zs = zs.cuda()
            fake_labels = fake_labels.cuda()

        fake_images = self.G(zs, fake_labels)
        if self.conditional_strategy == "ContraGAN":
            fake_cls_mask = make_mask(fake_labels, self.num_classes, mask_negatives=self.mask_negatives, device=self.device)
            cls_proxies_fake, cls_embed_fake, dis_out_fake = self.D(fake_images, fake_labels)

        
        if self.selected_loss in ['wasserstein', 'hinge']:
            g_loss =  -torch.mean(dis_out_fake.view(-1))
            
        if self.conditional_strategy == "ContraGAN":
            t = 1.0
            margin = 0.0
            contrastive_lambda = 1.0
            g_loss += contrastive_lambda*self.contrastive_criterion(cls_embed_fake, cls_proxies_fake, fake_cls_mask, fake_labels, t, margin)

        # elif self.selected_loss == 'classic':
        #     # self.label.fill_(1.)
        #     # print(g_fake_sigmoid.view(-1).shape)
        #     g_loss = self.BCE_loss(g_fake_sigmoid.view(-1), self.label_one)

        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['LossG'].append(g_loss.item())


    def _gradient_penalty(self, real_data, generated_data):
        ''' Compute gradient penalty '''

        # Compute interpolation
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)    
        interpolated = Variable( alpha * real_data + (1 - alpha) * generated_data, requires_grad=True).to(self.device)

        # Compute probability of interpolated examples
        _, logit_interpolated = self.D(interpolated)
            
        out_interpolated_batch = logit_interpolated.size()
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=logit_interpolated, inputs=interpolated,
                    grad_outputs=torch.ones(out_interpolated_batch).to(self.device) if self.use_cuda else torch.ones(
                     out_interpolated_batch),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradients_norm - 1) ** 2)


        # Return gradient penalty
        return self.gp_weight * gradient_penalty


    def _train_epoch(self, data_loader):
        self.G.train()
        # start_time = time.time()
        for i, data in enumerate(data_loader):
            
            # Get generated data
            img = data[0].to(self.device)
            labels = data[1].to(self.device)
            
            batch_size = img.size(0)
                
            # Prepare labels for classic loss
            if self.selected_loss == 'classic':
                self.label_one = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
                self.label_zero = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)
            self.num_steps += 1

            
            # Update Discriminator (excluding rotated generated images)
            self._critic_train_iteration(img, labels, batch_size)
            
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(batch_size)
                
            # Print Loss informations and plot generated images
            if self.num_steps % self.print_every == 0:
                # print('{} minutes'.format((time.time() - start_time)/60))
                
                print()
                print("Iteration {}".format(self.num_steps))
                print("Total Loss D: {}".format(self.losses['LossD'][-1]))
                    
                if len(self.losses['LossG']) !=0:
                    print("Total Loss G: {}".format(self.losses['LossG'][-1]))
                if self.gp_weight !=0:
                    print("GP: {}".format(self.losses['GP'][-1]))
                
                    # print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                print()
                # Plot images to track performance
                if self.plot_images: 
                    gen_imgs = self.genImages()
                    self.plotImages(gen_imgs)
                    
   
        self.EpochUpdateInfo() # Update informations
        self.DumpInfo() # save information about FID, Iterations...
        if self.saveModelsPerEpoch:
            self.save_model(self.checkpoint_folder) # save model generator and discriminator
        
        # if (self.epoch+1) % 2 == 0:
        #     if self.save_images: 
        #         gen_images = self.genImages()
        #         self.saveImages(gen_images,self.FIDPaths[0]) # save images per epoch to trace performance       
        

        if self.save_images: 
            gen_images = self.genImages()
            self.saveImages(gen_images, self.FIDPaths[0]) # save images per epoch to trace performance       


    def train(self, data_loader, epochs):
        ''' Train the network \n input: dataloader, epochs'''
        # self.fixed_noise = self.G.sample_latent(9).to(self.device)
        
        # print(self.G)
        # print()
        # print(self.D)

        self.fixed_noise, self.fixed_fake_labels = sample_latents(self.prior, 9, self.z_size, -1.0, self.num_classes, None)
        if self.conditional_strategy == 'ContraGAN':
            self.mask_negatives=True
            self.contrastive_criterion = Conditional_Contrastive_loss(self.batch_size, self.pos_collected_numerator, self.device)

        
        for epoch in range(self.epoch+1,epochs):
            print("\nEpoch {}".format(epoch))
            self.epoch = epoch
            start_time = time.time()
            self._train_epoch(data_loader)
            print('Epoch {} finished in {} minutes'.format(self.epoch, (time.time()-start_time)/60) )
        

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data

        
    def EpochUpdateInfo(self):
        '''Update LossD and LossG in tracked_info'''
        self.tracked_info['Epochs'] = self.epoch +1
        for k, v in self.losses.items():
            if k in self.tracked_info.keys():
                self.tracked_info[k] = v
                
        self.tracked_info['Iterations'] = self.num_steps
        
        # if self.save_FID==True:
        #     print('Calculate FID on {} Images'.format(self.FIDImages))
        #     tracked_FID = self.GenImgGetFID(self.FIDImages)
        #     self.tracked_info['FID'].append(tracked_FID)
        
        # if len(self.Fid_Values ) !=0:
        #     self.tracked_info['FID'].append( self.Fid_Values)
        # print(self.G.type)
            
        
    def DumpInfo(self):
        '''Dump tracked info'''
        if not os.path.isdir('./infos'):
            os.makedirs('./infos')
        info_path = './infos/'+ self.Generator_info + '_' + date + '.json'
        with open(info_path, 'w') as f:
          json.dump(self.tracked_info, f)
          
          
    def save_model(self, folder):
        '''Save D and G models \nfolder: where to save models'''
        
        gen_name = self.G.__class__.__name__
        disc_name = self.D.__class__.__name__
        
        gen_path = str(folder) + str(gen_name) +'/'

        if not os.path.isdir(gen_path):
          os.makedirs(gen_path)
          
        disc_path = str(folder) + str(disc_name) +'/'

        if not os.path.isdir(disc_path):
          os.makedirs(disc_path)
        
        varG={'epoch': self.epoch,
            'model_state_dict': self.G.state_dict(),
            'optimizer_state_dict': self.G_opt.state_dict(),
            'lossG': self.losses['LossG'],
	    'lossGP':self.losses['GP'],
            }
        torch.save(varG, gen_path + '_epoch{}'.format(self.epoch) + '_GSN-{}_DSN-{}_{}'.format(
            hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm ==True,
            hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm ==True, date) + '.pt')
        
        model_save_name = '_epoch{}'.format(self.epoch) + '_GSN-{}_DSN-{}_{}'.format(hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm ==True,
            hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm ==True, date) + '.pt'
        path = '/content/gdrive/My Drive/elemodelG/'+ model_save_name 
        torch.save(varG, path)

        varD={'epoch': self.epoch,
            'model_state_dict': self.D.state_dict(),
            'optimizer_state_dict': self.D_opt.state_dict(),
            'loss': self.losses['LossD'],
            }

        torch.save(varD, disc_path + '_epoch{}'.format(self.epoch) + '_GSN-{}_DSN-{}_{}'.format(
            hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm ==True,
            hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm ==True, date) + '.pt')
        
        model_save_name = '_epoch{}'.format(self.epoch) + '_GSN-{}_DSN-{}_{}'.format(hasattr(self.G, 'g_spectral_norm') and self.G.g_spectral_norm ==True,
            hasattr(self.D, 'd_spectral_norm') and self.D.d_spectral_norm ==True, date) + '.pt'
        path = '/content/gdrive/My Drive/elemodelD/'+ model_save_name 
        torch.save(varD, path)
            

    def genImages(self):
        ''' Return a 3x3 grid of 9 images'''
        with torch.no_grad():
            self.G.eval()
            
            if self.use_cuda:
                self.fixed_noise = self.fixed_noise.cuda()
                self.fixed_fake_labels = self.fixed_fake_labels.cuda()

            fake = self.G(self.fixed_noise, self.fixed_fake_labels)
            
            if fake.size(1) == 3:
                fake = fake
            else:
                fake = fake[:,1:4,:,:]
            
            if self.normalize:
                imgs = vutils.make_grid(fake.detach().cpu().data, normalize=True, range=(-1,1), padding=2, nrow=3)
            else:
                imgs = vutils.make_grid(fake.detach().cpu().data, padding=2, nrow=3)
        self.G.train()
        
        return imgs
    
        
    def plotImages(self, imgs):
        '''Plot grid of images'''
        if not hasattr(self, 'fig'):
            self.fig= '_'
            plt.figure(figsize=(3,3))
        plt.ion()
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(imgs.permute(1,2,0))
        plt.show()
        plt.pause(0.1)
            
        
    def saveImages(self, imgs, path):
        '''Save grid of images'''

        print('Images saved')
        plt.ioff()
        plt.axis("off")
        plt.title("Generated Images")
        imgs = imgs.permute(1,2,0)
        # images from [-1, 1] to [0, 1]
        # imgs = (imgs + 1) / 2
        img_path = path + '/Epoch{}'.format(self.epoch) + '.png'
        plt.imsave(img_path, imgs.numpy())
        # plt.close()
                
