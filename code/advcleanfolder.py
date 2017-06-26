import torch.utils.data as data
from PIL import Image
import os
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_dataset(clean_dir, adv_dir, idx = None):
    images = []
   
    if idx is None:
        idx = range(len(os.listdir(clean_dir)))
    else:
        pass
        #assert(len(os.listdir(clean_dir)) == len(os.listdir(adv_dir)))
        
    #for img_name in np.sort(os.listdir(clean_dir))[idx]:
    for i in idx:
        img_name = '%d.png' % i
        path = os.path.join(clean_dir, img_name)
        item = (path, 0)
        images.append(item)

    if idx is None:
        idx = range(len(os.listdir(adv_dir)))
    
    #for img_name in np.sort(os.listdir(adv_dir))[idx]:
    for i in idx:
        img_name = '%d.png' % i
        path = os.path.join(adv_dir, img_name)
        item = (path, 1)
        images.append(item)

    return images

class AdvCleanFolder(data.Dataset):
    def __init__(self, clean_dir, adv_dir, idx=None, transform=None, target_transform=None,
                 loader=pil_loader):
        imgs = make_dataset(clean_dir, adv_dir, idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
