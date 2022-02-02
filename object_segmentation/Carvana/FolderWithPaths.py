# -- coding: utf-8 --
"""
Created on Thu May 21 09:28:02 2020
@author: arun
"""
import os
from torchvision import datasets
from torch.utils.data import Dataset

class FolderWithPaths(Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def _init_(self, root, transform=None):
        self.image_dir = root
        self.imgs = [image_path for image_path in os.listdir(root)]
        
    def _len_(self):
        return len(self.imgs)
    
    # override the _getitem_ method. this is the method that dataloader calls
    def _getitem_(self, index):
        img_path = self.imgs[index]
        
        # return (index,self.image_dir, os.path.join(self.image_dir, img_path))
        return os.path.join(self.image_dir, img_path)