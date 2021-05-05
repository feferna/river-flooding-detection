import os
import torch
import cv2

import torchvision
import torchvision.transforms

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

### This code is based on the one found in https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class RiverDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, transform_target):
        self.root = root
        self.transform = transform
        self.transform_target = transform_target
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = self.transform(img)
        target = self.transform_target(target)

        target = np.array(target, dtype=np.int64)
        target = torch.as_tensor(target, dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PennFudanDataset(object):
    def __init__(self, root, transform, transform_target):
        self.root = root
        self.transform = transform
        self.transform_target = transform_target
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to Semantic Segmentation instead of Instance Segmentation
        target = np.where(target > 0, 1, 0).astype('uint8')

        img = self.transform(img)
        target = self.transform_target(target)

        target = np.array(target, dtype=np.int64)
        target = torch.as_tensor(target, dtype=torch.int64)
        return img, target

    def __len__(self):
        return len(self.imgs)


# class PascalVOC(object):
#     def __init__(self, root, transform, transform_target):
#         self.root = root
#         self.transform = transform
#         self.transform_target = transform_target
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

#     def __getitem__(self, idx):
#         # load images ad masks
#         img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
#         # Convert to Semantic Segmentation instead of Instance Segmentation
#         target = np.where(target > 0, 1, 0).astype('uint8')

#         img = self.transform(img)
#         target = self.transform_target(target)

#         target = np.array(target, dtype=np.int64)
#         target = torch.as_tensor(target, dtype=torch.int64)
#         return img, target

#     def __len__(self):
#         return len(self.imgs)


### This code is based on the one found in https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def data_loader(dataset_name, batch_size):
    if dataset_name == "SaoCarlos":
        #normalize = torchvision.transforms.Normalize(mean=[0.53746605, 0.5228257, 0.46264213], std=[0.19829234, 0.18723851, 0.19410051])
        normalize = torchvision.transforms.Normalize(mean=[0.5291628, 0.5135074, 0.45702627], std=[0.19916107, 0.18686345, 0.1919754])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            normalize])

        transform_target = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=(224, 224))])

        # use our dataset and defined transformations
        #dataset = RiverDataset("./floodwall_dataset_no_floodwall_v1.3", transform, transform_target)
        dataset = RiverDataset("./floodwall_dataset_no_floodwall_v1.2", transform, transform_target)
        dataset_test = RiverDataset("./floodwall_dataset_test_set", transform, transform_target)

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)

    elif dataset_name == "PennFudanPed":
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            normalize])

        transform_target = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=(224, 224))])

        # use our dataset and defined transformations
        dataset = PennFudanDataset('../Datasets/PennFudanPed', transform, transform_target)

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=11)

        dataset_train = torch.utils.data.Subset(dataset, train_idx)
        dataset_test = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)
    
    elif dataset_name == "PascalVOC":
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        transform = torchvision.transforms.Compose([
            #torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            normalize])

        transform_target = torchvision.transforms.Compose([
            #torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224, interpolation=Image.NEAREST),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.ToTensor()
        ])

        # use our dataset and defined transformations
        dataset = torchvision.datasets.VOCSegmentation("../Datasets/PascalVOC/", "2012", "train", False, transform, transform_target)
        #dataset = PennFudanDataset('../Datasets/PennFudanPed', transform, transform_target)

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=11)

        dataset_train = torch.utils.data.Subset(dataset, train_idx)
        dataset_test = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            num_workers=6,
            pin_memory=True)

    return train_loader, test_loader
