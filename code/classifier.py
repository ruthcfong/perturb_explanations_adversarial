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
from helpers import AverageMeter, save_state, GetFirstChannel, adjust_learning_rate, accuracy, run_epoch, read_imdb
from advcleanfolder import AdvCleanFolder
from custom_alexnet import CustomAlexNet
from defaults import mus, sigmas

class CustomNet(nn.Module):

    def __init__(self, num_classes=2, num_input_channels=1):
        super(CustomNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 6 * 6)
        x = self.classifier(x)
        return x

def main():
    batch_size = 128
    num_workers = 4
    gpu = 0 

    cuda = True if gpu is not None else False
    use_mult_gpu = isinstance(gpu, list)
    if use_mult_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
    print torch.cuda.device_count(), use_mult_gpu, cuda


    conserve = True # conserve memory, don't save state
    heatmap_mode = 'single'
    all_heatmap_types = ['mask', 'saliency', 'guided_backprop', 'grad_cam', 'contrast_excitation_backprop', 'excitation_backprop']
    normalize = True 
    num_iters = 50
    if heatmap_mode == 'all':
        heatmap_types = all_heatmap_types
    elif heatmap_mode == 'all_but_one':
        excluded_heatmap = 'mask'
        heatmap_types = [h for h in all_heatmap_types if h is not excluded_heatmap]
    elif heatmap_mode == 'single':
        heatmap_type = 'image'
        if heatmap_type == 'mask':
            folder_name = 'defaults_iter_%d' % num_iters
        elif heatmap_type == 'image':
            pretrained = False 
            folder_name = '%s_pretrained_%d' % (heatmap_type, pretrained)
            normalize = True
        else:
            folder_name = heatmap_type

    #adv_types = ['fgsm', 'one_step', 'fgsm_iter', 'one_step_iter']
    adv_types = ['one_step_iter']
    #epsilons = [8]
    epsilons = [1,2,4,8,12]

    for eps in epsilons:
        for adv_type in adv_types:
            #if os.path.exists(fn):
            #    print '%s exists so skipping' % fn
            #    continue

            if heatmap_mode == 'single':
                if heatmap_type is 'image':
                    model = CustomAlexNet(2, 3, pretrained=pretrained)
                else:
                    model = CustomAlexNet(2, 1)
            else:
                model = CustomAlexNet(2, len(heatmap_types))

            #model = CustomNet(2,1)
            if use_mult_gpu and cuda:
                model.features = torch.nn.DataParallel(model.features)
            if cuda:
                model.cuda()

            if heatmap_mode == 'single':
                if heatmap_type == 'image':
                    transform = transforms.Compose([
                        transforms.Scale(227, interpolation=Image.BILINEAR),
                        transforms.CenterCrop(227),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                             std = [ 0.229, 0.224, 0.225 ]),
                    ])
                else:
                    if normalize:
                        mu = mus[folder_name]
                        sigma = sigmas[folder_name]
                        transform = transforms.Compose([
                            transforms.Scale(227, interpolation=Image.NEAREST),
                            transforms.CenterCrop(227),
                            transforms.ToTensor(),
                            transforms.Normalize((mu, mu, mu), (sigma, sigma, sigma)),
                            GetFirstChannel()
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Scale(227, interpolation=Image.NEAREST),
                            transforms.CenterCrop(227),
                            transforms.ToTensor(),
                            GetFirstChannel()
                        ])
            else:
                if normalize:
                    transform = []
                    for h in heatmap_types:
                        if h == 'mask':
                            folder_name = 'defaults_iter_%d' % num_iters
                        else:
                            folder_name = h
                        mu = mus[folder_name]
                        sigma = sigmas[folder_name] 
                        t = transforms.Compose([
                            transforms.Scale(227, interpolation=Image.NEAREST),
                            transforms.CenterCrop(227),
                            transforms.ToTensor(),
                            transforms.Normalize((mu, mu, mu), (sigma, sigma, sigma)),
                            GetFirstChannel()
                        ])
                        transform.append(t)
                else:
                    transform = transforms.Compose([
                        transforms.Scale(227, interpolation=Image.NEAREST),
                        transforms.CenterCrop(227),
                        transforms.ToTensor(),
                        GetFirstChannel()
                    ])


            if heatmap_mode == 'single':
                fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/checkpoint/custom_alexnet_%s_eps_%d_%s_norm_%d_checkpoint.pth.tar' % (
                        adv_type, eps, folder_name, normalize)
                small_fn = '/data/ruthfong/perturb_explanations_adversarial/results_classifiers/custom_alexnet_%s_eps_%d_%s_norm_%d_checkpoint.pth.tar' % (
                        adv_type, eps, folder_name, normalize)
                best_fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/best/custom_alexnet_%s_eps_%d_%s_norm_%d_best.pth.tar' % (
                        adv_type, eps, folder_name, normalize)
                if heatmap_type == 'image':
                    (clean_dir, _) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt')
                    adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_train_heldout/%s/eps_%d' % (adv_type, eps) 
                else:
                    clean_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s' % folder_name 
                    adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/%s' % (
                            adv_type, eps, folder_name)
            else:
                if heatmap_mode == 'all':
                    fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/checkpoint/custom_alexnet_%s_eps_%d_%s_norm_%d_checkpoint.pth.tar' % (
                            adv_type, eps, heatmap_mode, normalize)
                    small_fn = '/data/ruthfong/perturb_explanations_adversarial/results_classifiers/custom_alexnet_%s_eps_%d_%s_norm_%d_checkpoint.pth.tar' % (adv_type, eps, heatmap_mode,normalize)
                    best_fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/best/custom_alexnet_%s_eps_%d_%s_norm_%d_best.pth.tar' % (
                            adv_type, eps, heatmap_mode,normalize)
                elif heatmap_mode == 'all_but_one':
                    fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/checkpoint/custom_alexnet_%s_eps_%d_%s_%s_norm_%d_checkpoint.pth.tar' % (adv_type, eps, heatmap_mode, excluded_heatmap, normalize)
                    small_fn = '/data/ruthfong/perturb_explanations_adversarial/results_classifiers/custom_alexnet_%s_eps_%d_%s_%s_norm_%d_checkpoint.pth.tar' % (adv_type, eps, heatmap_mode, excluded_heatmap, normalize)
                    best_fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/best/custom_alexnet_%s_eps_%d_%s_%s_norm_%d_best.pth.tar' % (
                            adv_type, eps, heatmap_mode, excluded_heatmap, normalize)

                clean_dir = []
                adv_dir = []
                for h in heatmap_types:
                    if h == 'mask':
                        num_iters = 50
                        folder_name = 'defaults_iter_%d' % num_iters
                    else:
                        folder_name = h 
                    clean_dir.append('/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s' % folder_name)
                    adv_dir.append('/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/%s' % (
                        adv_type, eps, folder_name))
            train_f = AdvCleanFolder(clean_dir, adv_dir, idx=range(100,4000), transform=transform)
            val_f = AdvCleanFolder(clean_dir, adv_dir, idx=range(4000,5000), transform=transform)

            train_loader = torch.utils.data.DataLoader(train_f, batch_size=batch_size, pin_memory=True, num_workers=num_workers, 
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_f, batch_size=batch_size, pin_memory=True, num_workers=num_workers,
                                                       shuffle=True)

            #starting_lr = 1e-2
            #momentum = 0.9
            #weight_decay = 1e-4
            #optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=momentum, weight_decay=weight_decay)
            optimizer = torch.optim.Adam(model.parameters())

            print_freq = 25

            min_num_epochs = 25 
            best_loss = np.inf
            best_acc = 0
            best_epoch = -1
            criterion = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()

            avg_trn_losses = []
            avg_trn_accs = []
            avg_val_losses = []
            avg_val_accs = []

            for epoch in range(100):
                start = time.time()

                #adjust_learning_rate(optimizer, epoch, starting_lr, freq = 25)

                # train for an epoch
                (avg_trn_loss, avg_trn_acc) = run_epoch(train_loader, model, criterion, optimizer, epoch, train = True, cuda = cuda,
                    print_freq = print_freq, save_freq = None)

                (avg_val_loss, avg_val_acc) = run_epoch(val_loader, model, criterion, optimizer, epoch, train = False, cuda = cuda,
                    print_freq = print_freq, save_freq = None)

                avg_trn_losses.append(avg_trn_loss)
                avg_trn_accs.append(avg_trn_acc)
                avg_val_losses.append(avg_val_loss)
                avg_val_accs.append(avg_val_acc)

                # save state
                is_best = avg_val_loss < best_loss
                best_loss = min(avg_val_loss, best_loss)
                if is_best:
                    best_acc = avg_val_acc
                    best_epoch = epoch

                print 'epoch time', time.time()-start, 'is_best', is_best, 'best_loss', best_loss, 'best_acc', best_acc, 'best_epoch', best_epoch, 'curr_loss', avg_val_loss, 'curr_acc', avg_val_acc

                state = {'epoch': epoch+1,
                                'state_dict': model.state_dict(),
                                'best_loss': best_loss,
                                'best_acc': best_acc,
                                'best_epoch': best_epoch,
                                'curr_loss': avg_val_loss,
                                'curr_acc': avg_val_loss,
                                'trn_losses': avg_trn_losses,
                                'trn_accs': avg_trn_accs,
                                'val_losses': avg_val_losses,
                                'val_accs': avg_val_accs,
                                }
                if not conserve:
                    save_state(fn, state, best_fn = best_fn, is_best = is_best)
                state = {key: value for key, value in state.items() 
                                     if key != 'state_dict'}
                save_state(small_fn, state)

                if epoch-best_epoch > min_num_epochs:
                    print 'stopping early epoch ', epoch+1
                    break

if __name__ == '__main__':
    main()
