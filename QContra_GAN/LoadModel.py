# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:39:50 2020

@author: Edoardo
"""


import torch
from GetModel import GetModel
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision import utils as vutils
# import numpy as np
# from torchvision.utils import save_image
# import torchvision
import os

import sys
sys.path.append('./utils')
from utils.pytorch_fid.fid_score import FID
from utils.inception_score import inception_score
from utils.Qdataloaders import CIFAR10_dataloader

from utils.sample import sample_latents
# from utils.pytorch_fid import fid_score

check_path_G = "./ch/MidQuaternionGenerator/_epoch1_GSN-False_DSN-True_2021-09-09_09-52-57.pt"

model = 'MidQDCGAN'
model_name = 'MidQDCGAN'

CelebAHQ = False
if CelebAHQ:
    print('Dataset: CelebA HQ')
else:
    print('Dataset: CIFAR')

use_cuda = True
BN = True
g_spectral_norm = True
d_spectral_norm = True
batch_size = 50
eval_mode = True
z_size = 80

images = 10000
normalize = True

plott = False
save_imgs = True
getFid = True

print('Eval mode:',eval_mode)
epoch = (check_path_G.split('/')[-1]).split('_')[2]
save_path = './results/Imgs_FID_TEST/'

if not CelebAHQ:
    print('on CIFAR')
    CIFAR10_dataloader(root=None, quat_data=True, img_size=32, normalize=True, batch_size=batch_size, num_workers=1, eval=True)
    original = './data/Test_FID_cifar/test_cifar'  
else:
    print('on CelebA-HQ')
    original = '/var/datasets/Test_FID_3000_HQ/Test_FID_3000_HQ'

#original = './data/Test+Train_FID_10000_HQ/'
print("Real Images for FID:", len(os.listdir(original)))
print('Images generated:', str(images))


if not normalize:
    save_path += 'NotNormalized/'

if eval_mode:
    save_path += model_name + '_' + epoch + '_' + 'GSN-' + str(g_spectral_norm) + '_' + 'DSN-' + str(d_spectral_norm) + '_EVAL' + "/"
else:
    save_path +=  model_name + '_' + epoch + "/"
print('Save_path:', save_path)


if plott or save_imgs:
    
    if save_imgs:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    
    # info = check_path.split()
    # checkpoint_D = torch.load(check_path_D, map_location='cuda:0')
    checkpoint_G = torch.load(check_path_G, map_location='cuda:0')
    modelG, modelD = GetModel(str_model=model, z_size=z_size, img_size=32, g_conv_dim=96, d_conv_dim=96,
                                g_spectral_norm=g_spectral_norm, d_spectral_norm=d_spectral_norm, attention=True,
                                attention_after_nth_gen_block=2, attention_after_nth_dis_block=1, conditional_strategy="ContraGAN",
                                num_classes=10, hypersphere_dim=512, nonlinear_embed=False, normalize_embed=True, needs_init=True, mixed_precision=False)

    modelG.to('cuda:0')
    modelD.to('cuda:0')
    # print(checkpoint_D)
    modelG.load_state_dict(checkpoint_G)
    # modelD.load_state_dict(checkpoint_D)
             
                
 
    with torch.no_grad():
        G = modelG
        if eval_mode:
            G.eval()
            
        for i in range(images):
            
            zs, fake_labels  = sample_latents("gaussian", 1, z_size, -1.0, 10, None)

            if use_cuda:
                zs = zs.cuda()
                fake_labels = fake_labels.cuda()

            fake = G(zs, fake_labels)

            if fake.size(1) == 3:
                fake = fake[0].permute(1,2,0)
            else:
                fake = fake[0,1:4,:,:].permute(1,2,0)
                
            if normalize:
                fake = vutils.make_grid(fake.cpu().data, normalize=True, range=(-1,1))
            else:
                fake = vutils.make_grid(fake.cpu().data)
                
            if plott:
                plt.imshow(fake.cpu())
                plt.show()#block=False)
                
            if save_imgs:
                if not normalize:
                    fake = torch.clamp(fake, 0, 1)
                

                plt.imsave(save_path + 'IMG{}.png'.format(i), fake.cpu().numpy()  )

if getFid:
    dset = datasets.ImageFolder(root='./results', transform=transforms.ToTensor())
    print(inception_score(dset, len(dset), cuda=True, batch_size=32, resize=True, splits=10))

    print('Computing FID...')
    
    print('path:', original)
    fid = FID()(path=[save_path, original])
    # fid = fid_score.main(device='cuda:0', path=['C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/results/Imgs_FID_TEST/', 'C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/data/Test_FID_3000_HQ/Test_FID_3000_HQ'], batch_size=50, dims=2048)
    print(fid)
    
