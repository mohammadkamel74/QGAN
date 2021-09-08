# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:24:26 2020

@author: Edoardo
"""

import argparse 
import torch
import torch.optim as optim
# from utils.Qdataloaders import get_CelebA_QDCGAN_dataloader, get_CelebA_DCGAN_dataloader
from utils.Qdataloaders import  CelebA_dataloader2, CelebA_colab_dataloader, CelebAHQ_dataloader, LSUN_dataloader
from utils.Qdataloaders import CIFAR10_dataloader

# from Qmodel import Generator, Discriminator, Simple_Discriminator, Simple_Generator
# from Qmodel import DCGAN_Generator, DCGAN_Discriminator
# from models.QRotGAN_32 import QRotGAN_G32, QRotGAN_D32
# from models.QRotGAN_128 import QRotGAN_G128, QRotGAN_D128
# from models.Q_DCGAN_64 import DCGAN_Discriminator, DCGAN_Generator, QDCGAN_Discriminator, QDCGAN_Generator
# from models.Test_QGAN import QGANLastConv_D, QGANLastConv_G

from Qtraining_new import Trainer
# from torch import nn
import random
import numpy as np
import os
from GetModel import GetModel
from utils.readFile import readFile
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()#fromfile_prefix_chars='@')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--colab', type=bool, default=False)
    parser.add_argument('--n_workers', default=1)
    
    parser.add_argument('--train_dir', type=str, default='./data/celebA_Train/Train', help="Folder containg training data. It must point to a folder with images in it.")
    
    parser.add_argument('--Dataset', type=str, default='CelebA_GAN', help='CelebA_GAN, CelebAHQ_GAN')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--normalize', type=bool, default=False, help='map value of images from range [0,255] to range [-1,1]')
    
    parser.add_argument('--model', type=str, default='DCGAN', help='Models: DCGAN')

    parser.add_argument('--z_size', type=int, default=80)
    parser.add_argument('--BN', type=bool, default=False, help='Apply Batch Normalization')
    
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    
    parser.add_argument('--loss', type=str, default='hinge', help='[hinge, classic, wasserstein]')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--betas', default=(0.5, 0.999))
    
    parser.add_argument('--crit_iter', type=int, default=1, help='critic iteration') 
    parser.add_argument('--gp_weight', type=int, default=0, help='[1,10] for SSGAN, default=0')
    
    parser.add_argument('--print_every', type=int, default=50, help='Print Gen and Disc Loss every n iterations')
    parser.add_argument('--plot_images', type=bool, default=True, help='Plot images during training')
    parser.add_argument('--save_images', type=bool, default=True, help='Save images every epoch to track performance')
    parser.add_argument('--EpochCheckpoints', type=bool, default=True, help='Save model every epoch. If set to False the model will be saved only at the end')
    
    parser.add_argument('--save_FID', type=bool, default=True, help='Save images and compute FID score')
    parser.add_argument('--Test_FID_dir', type=str, default='./data/celeba/img_align_celeba/Test_FID_100/', help='Path to Folder with Test images for FID')

    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')

    # add on 03-08

    parser.add_argument('--prior', type=str, default='gaussian', help='')    
    parser.add_argument('--g_conv_dim', type=int, default=96, help='')
    parser.add_argument('--d_conv_dim', type=int, default=96, help='')
    parser.add_argument('--g_spectral_norm', type=bool, default=False, help='Apply Spectral Normalization to Generator')
    parser.add_argument('--d_spectral_norm', type=bool, default=False, help='Apply Spectral Normalization to Discriminator')
    parser.add_argument('--attention', type=bool, default=True, help='Apply attention')
    parser.add_argument('--attention_after_nth_gen_block', type=int, default=2, help='Apply attention after block Generator')
    parser.add_argument('--attention_after_nth_dis_block', type=int, default=1, help='Apply attention after block Discriminator')
    parser.add_argument('--conditional_strategy', type=str, default='ContraGAN', help='Type of conditional strategy to apply')
    parser.add_argument('--num_classes', type=int, default=10, help='')
    parser.add_argument('--hypersphere_dim', type=int, default=512, help='')
    parser.add_argument('--nonlinear_embed', type=bool, default=False, help='')
    parser.add_argument('--normalize_embed', type=bool, default=True, help='')
    parser.add_argument('--mixed_precision', type=bool, default=False, help='')
    parser.add_argument('--pos_collected_numerator', type=bool, default=True, help='')
    parser.add_argument('--needs_init', type=bool, default=True, help='')
    parser.add_argument('--freeze_layers', type=int, default=-1, help='')

    parse_list=readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)
    
    use_cuda = opt.cuda
    gpu_num = opt.gpu_num
    loss = opt.loss
    critic_iterations = opt.crit_iter  # [1, 2]
    gp_weight = opt.gp_weight         # [1, 10]
    
    lr = opt.lr
    betas = opt.betas.replace(',', ' ').split()
    betas = (float(betas[0]), float(betas[1]))
    # print(betas)
    epochs = opt.epochs
    BN = opt.BN # Batch normalization
    # print(BN)
    
    save_FID = opt.save_FID
    plot_images = opt.plot_images
    
    freeze_layers = opt.freeze_layers
    pos_collected_numerator = opt.pos_collected_numerator
    prior = opt.prior
    z_size = opt.z_size
    g_conv_dim = opt.g_conv_dim
    d_conv_dim = opt.d_conv_dim
    g_spectral_norm = opt.g_spectral_norm
    d_spectral_norm = opt.d_spectral_norm
    attention = opt.attention
    attention_after_nth_gen_block = opt.attention_after_nth_gen_block
    attention_after_nth_dis_block = opt.attention_after_nth_dis_block
    conditional_strategy = opt.conditional_strategy
    num_classes = opt.num_classes
    hypersphere_dim = opt.hypersphere_dim
    nonlinear_embed = opt.nonlinear_embed
    normalize_embed = opt.normalize_embed
    needs_init = opt.needs_init
    mixed_precision = opt.mixed_precision
    img_size = opt.image_size
    batch_size = opt.batch_size
    print_every = opt.print_every
    EpochCheckpoints = opt.EpochCheckpoints
    save_images = opt.save_images
    
    dataset = opt.Dataset
    colab = opt.colab
    n_workers = opt.n_workers
    
    if n_workers=='max':
        n_workers = cpu_count()
    
    
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
    model = opt.model
    train_dir = opt.train_dir
    normalize = opt.normalize

    generator, discriminator = GetModel(model, z_size, img_size, g_conv_dim, d_conv_dim,
                                        g_spectral_norm, d_spectral_norm, attention, attention_after_nth_gen_block,
                                        attention_after_nth_dis_block, conditional_strategy, num_classes, hypersphere_dim,
                                        nonlinear_embed, normalize_embed, needs_init, mixed_precision)
    
    G_params= sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print('G parameters:', G_params)
    D_params= sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('D parameters:', D_params)
    print('Total parameters:', G_params+D_params)
    print()
    
    if 'Q' in generator.__class__.__name__:
        quat_data = True
    else:
        quat_data= False
            
    if dataset == 'CelebA_GAN':
        if not colab:
            data_loader, _ , data_name = CelebA_dataloader2(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        else:
            data_loader, _ , data_name = CelebA_colab_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
    
    elif dataset == 'CelebAHQ_GAN':
        data_loader, _ , data_name = CelebAHQ_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        
    elif dataset == 'CIFAR10':
        data_loader, _ , data_name = CIFAR10_dataloader(root=train_dir, quat_data=quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)

    else:
        RuntimeError('Wrong dataset or not implemented')
    
    gen_img_path = './generated_images/'
    real_img_path = opt.Test_FID_dir
    FID_paths = [gen_img_path, real_img_path]
    
    
    checkpoint_folder = './ch/'
    #if not os.path.isdir(checkpoint_folder):
    #    os.makedirs(checkpoint_folder)
    
    # Initialize optimizers
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    '''Train model'''
    trainer = Trainer(z_size, prior, num_classes, pos_collected_numerator, freeze_layers, conditional_strategy, batch_size,
                      generator, discriminator, G_optimizer, D_optimizer,
                      use_cuda=use_cuda, gpu_num=gpu_num, print_every = print_every,
                      loss = loss,
                      gp_weight=gp_weight,
                      critic_iterations=critic_iterations,
                      save_FID = save_FID,
                      FIDPaths = [gen_img_path, real_img_path],
                      checkpoint_folder = checkpoint_folder,
                      plot_images=plot_images,
                      save_images=save_images,
                      saveModelsPerEpoch=EpochCheckpoints,
                      normalize = normalize,
                      )
    
    
    trainer.train(data_loader, epochs)
