import torch
import torchvision

from library.image import square_padding
import json
import numpy as np
__all__ = ['Designer']


class Designer(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_transform):
        self.image_transform = image_transform
        self.image_paths = image_paths

    def __getitem__(self, index):
        images = []
        if isinstance(self.image_paths[index],list):
            for img_pth in self.image_paths[index]:
                image = torchvision.io.read_image(str(img_pth), mode=torchvision.io.ImageReadMode.RGB)
                image = square_padding(image)
                image = self.image_transform(image)
                images.append(image)
            images = torch.stack(images,dim=0)
        else:
            images = torchvision.io.read_image(str(self.image_paths[index]), mode=torchvision.io.ImageReadMode.RGB)
            images = square_padding(images)
            images = self.image_transform(images)

        return images, index

    def __len__(self):
        return len(self.image_paths)



class DesignerSet(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 is_training=False,
                 data_augument=False,
                 image_transform=None,
                 mode='',
                 root='.'
                 ):
        if root is None or root=='':
            root='.'
        self.root = root
        self.args = args
        self.num_class = args.num_classes
        self.is_training = is_training
        self.data_augument = data_augument
        self.image_transform = image_transform
        self.cats=["dress","shoes","bag"]
        self.data = []
        json_file = root+"/"+mode+".json"
        with open(json_file, 'r') as json_file:
            self.data = json.load(json_file)


    def __getitem__(self, index):
        data = self.data[index]
        images = []
        for cat in self.cats:
            img_pth = self.root+"//"+data["brand"]+"//"+data["look_name"]+"//"+data[cat]
            image = self.pad_image(torchvision.io.read_image(img_pth,mode=torchvision.io.ImageReadMode.RGB))
            image = self.pad_image(image)
            image = self.image_transform(image)
            images.append(image)
        images = torch.stack(images,dim=0)
        label = data["label"]
        label_vec = np.zeros(self.num_class)
        label_vec[label]=1
        # return images,label_vec
        return images, label

    def __len__(self):
        return len(self.data)

    # Make all photos square
    def pad_image(self, img):
        h, w = img.shape[1:]
        if h != w:
            new_w = max(h, w)
            pad_h, rem_h = divmod(new_w - h, 2)
            pad_w, rem_w = divmod(new_w - w, 2)
            padding = [pad_w, pad_h, pad_w + rem_w, pad_h + rem_h]
            return torchvision.transforms.functional.pad(img, padding, padding_mode='edge')
        return img