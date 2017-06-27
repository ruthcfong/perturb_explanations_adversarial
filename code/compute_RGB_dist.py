import torch

import torchvision.transforms as transforms
import torch.utils.data as data
from helpers import pil_loader, AverageMeter,GetFirstChannel

import numpy as np
from PIL import Image
import os

def make_dataset(root, idx = None):
    images = []
    if isinstance(root, basestring):
        if idx is None:
            for f in os.listdir(root):
                if os.path.isdir(os.path.join(root, f)):
                    continue
                images.append(os.path.join(root, f))
                return images
        else:
            for i in idx:
                images.append((os.path.join(root, '%d.png' % i), 0))
    else:
        if idx is None:
            idx = range(len(root))
        for f in root[idx]:
            images.append(f)
    return images

     

class UnlabeledData(data.Dataset):
    def __init__(self, root, idx=None, transform=None, loader=pil_loader):
        self.imgs = make_dataset(root)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        #return (img, 0)
        return img

    def __len__(self):
        return len(self.imgs)


def main():
    batch_size = 128
    num_workers = 4

    heatmap_types = ['mask', 'saliency', 'guided_backprop', 'grad_cam', 'contrast_excitation_backprop', 'excitation_backprop']
    transform = transforms.Compose([
        transforms.Scale(227, interpolation=Image.NEAREST),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        GetFirstChannel(),
        ])
    
    for heatmap_type in heatmap_types:
        if heatmap_type == 'mask':
            num_iters = 50
            folder_name = 'defaults_iter_%d' % num_iters
        else:
            folder_name = heatmap_type
        clean_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s' % folder_name
        data = UnlabeledData(clean_dir, idx=range(100,4000), transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=num_workers,
                                                       shuffle=True)
        mu = AverageMeter()

        for i, input in enumerate(loader):
            #print input.size()
            mu.update(torch.mean(input.view(input.size(0),-1),1), input.size(0))

        mu = torch.squeeze(mu.avg).numpy()[0].astype(float)

        sigma_sq = AverageMeter()
        for i, input in enumerate(loader):
            sigma_sq.update(torch.div(torch.sum(torch.pow(input.view(input.size(0),-1)-mu, 2), 1), 227*227), input.size(0))
        sigma = np.sqrt(torch.squeeze(sigma_sq.avg).numpy()[0])

        print '%s: mean = %f, std = %f' % (heatmap_type, mu, sigma)


if __name__ == '__main__':
    main()
