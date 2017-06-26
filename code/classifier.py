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
from helpers import AverageMeter, save_state, GetFirstChannel, adjust_learning_rate, accuracy, run_epoch
from advcleanfolder import AdvCleanFolder
from custom_alexnet import CustomAlexNet

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
    gpu = 1 

    cuda = True if gpu is not None else False
    use_mult_gpu = isinstance(gpu, list)
    if use_mult_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
    print torch.cuda.device_count(), use_mult_gpu, cuda

    heatmap_type = 'mask'
    #adv_type = 'one_step_iter'
    #eps = 8
    num_iters = 50
    if heatmap_type == 'mask':
        folder_name = 'defaults_iter_%d' % num_iters
    else:
        folder_name = heatmap_type

    #adv_types = ['fgsm', 'one_step', 'fgsm_iter', 'one_step_iter']
    adv_types = ['fgsm_iter']
    epsilons = [4]

    for eps in epsilons:
        for adv_type in adv_types:
            fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/custom_alexnet_%s_eps_%d_%s_checkpoint.pth.tar' % (adv_type, 
                    eps, folder_name)
            best_fn = '/data/ruthfong/perturb_explanations_adversarial/classifiers/custom_alexnet_%s_eps_%d_%s_best.pth.tar' % (adv_type, eps, folder_name) 
            #if os.path.exists(fn):
            #    print '%s exists so skipping' % fn
            #    continue

            model = CustomAlexNet(2, 1)
            #model = CustomNet(2,1)
            if use_mult_gpu and cuda:
                model.features = torch.nn.DataParallel(model.features)
            if cuda:
                model.cuda()

            transform = transforms.Compose([
                transforms.Scale(227, interpolation=Image.NEAREST),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                GetFirstChannel()
                #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                #                     std = [ 0.229, 0.224, 0.225 ]),
            ])

            clean_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s' % folder_name 
            adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/%s' % (adv_type, eps, folder_name)
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
                save_state(fn, state, best_fn = best_fn, is_best = is_best)

                if epoch-best_epoch > min_num_epochs:
                    print 'stopping early epoch ', epoch+1
                    break

if __name__ == '__main__':
    main()
