import torch.utils.data as data
import torch
from PIL import Image
import os
import numpy as np
from helpers import pil_loader

def make_dataset(clean_dir, adv_dir, idx = None, is_single = True):
    images = []

    if idx is None:
        idx = range(len(os.listdir(clean_dir)))
    else:
        pass
        #assert(len(os.listdir(clean_dir)) == len(os.listdir(adv_dir)))
        
    #for img_name in np.sort(os.listdir(clean_dir))[idx]:
    for i in idx:
        img_name = '%d.png' % i
        if is_single:
            # account for 'image', where clean_dir is a list of image paths
            if isinstance(clean_dir, basestring):
                path = os.path.join(clean_dir, img_name)
            else:
                path = clean_dir[i]
            item = (path, 0)
        else:
            item = []
            for c_dir in clean_dir:
                path = os.path.join(c_dir, img_name)
                item.append(path)
            item.append(0)
            item = tuple(item)
        images.append(item)

    if idx is None:
        idx = range(len(os.listdir(adv_dir)))
    
    #for img_name in np.sort(os.listdir(adv_dir))[idx]:
    for i in idx:
        img_name = '%d.png' % i
        if is_single:
            path = os.path.join(adv_dir, img_name)
            item = (path, 1)
        else:
            item = []
            for a_dir in adv_dir:
                path = os.path.join(a_dir, img_name)
                item.append(path)
            item.append(1)
            item = tuple(item)
        images.append(item)

    return images

class AdvCleanFolder(data.Dataset):
    def __init__(self, clean_dir, adv_dir, idx=None, transform=None, target_transform=None,
                 loader=pil_loader):
        is_single = isinstance(adv_dir, basestring)
        imgs = make_dataset(clean_dir, adv_dir, idx, is_single)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.is_single = is_single
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        self.imgs = imgs
        self.classes = ['clean','adversarial']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.is_single:
            (path, target) = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            imgs = self.imgs[index]
            target = imgs[-1]
            img_i = self.loader(imgs[0])
            if self.transform is not None:
                if isinstance(self.transform, list):
                    img_i = self.transform[0](img_i)
                else:
                    img_i = self.transform(img_i)
            img = torch.zeros(len(imgs)-1, *torch.squeeze(img_i).size())
            for i in range(len(imgs)-1):
                img_i = self.loader(imgs[i])
                if self.transform is not None:
                    if isinstance(self.transform, list):
                        img_i = self.transform[0](img_i)
                    else:
                        img_i = self.transform(img_i)
                img[i] = img_i

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
