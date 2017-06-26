import shutil, os, sys, time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from helpers import GetFirstChannel, get_correctly_classified
from advcleanfolder import AdvCleanFolder
from custom_alexnet import CustomAlexNet

def main():
    batch_size = 128
    num_workers = 4
    gpu = 0 
    heatmap_type = 'grad_cam'

    cuda = True if gpu is not None else False
    use_mult_gpu = isinstance(gpu, list)
    if use_mult_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
    print torch.cuda.device_count(), use_mult_gpu, cuda

    state_path = '/data/ruthfong/perturb_explanations_adversarial/custom_alexnet_%s_best.pth.tar' % heatmap_type
    save_path = '/data/ruthfong/perturb_explanations_adversarial/custom_alexnet_%s_best_val_classification_indicator.txt' % heatmap_type

    model = CustomAlexNet(2, 1)
    #model = CustomNet(2, 1)
    if state_path is not None:
        model.load_state_dict(torch.load(state_path)['state_dict'])
    if use_mult_gpu and cuda:
        model.features = torch.nn.DataParallel(model.features)
    if cuda:
        model.cuda()

    transform = transforms.Compose([
        transforms.Scale(227, interpolation=Image.NEAREST),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        GetFirstChannel()
        #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        #                     std = [ 0.229, 0.224, 0.225 ]),
    ])

    clean_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s' % heatmap_type
    adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/one_step_iter/eps_4/%s' % heatmap_type
    #train_f = AdvCleanFolder(clean_dir, adv_dir, idx=range(4000), transform=transform)
    val_f = AdvCleanFolder(clean_dir, adv_dir, idx=range(4000,5000), transform=transform)

    #train_loader = torch.utils.data.DataLoader(train_f, batch_size=batch_size, pin_memory=True, num_workers=num_workers, 
    #                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_f, batch_size=batch_size, pin_memory=True, num_workers=num_workers,
                                               shuffle=False)

    print_freq = 1 

    indicator = get_correctly_classified(val_loader, model, cuda, print_freq = print_freq)
    np.savetxt(save_path, indicator)
    print 'saved to %s' % save_path
    indicator = np.array(indicator).reshape(2,1000)
    print 'clean acc: %.4f, adv acc: %.4f'% (indicator[0].sum()/float(len(indicator[0])), indicator[1].sum()/float(len(indicator[1])))


if __name__ == '__main__':
    main()
