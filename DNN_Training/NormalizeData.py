import torch
import numpy as np
from torchvision import transforms
import os
import cv2

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
        # river_mask = cv2.imread("./IMG_MASK.png", cv2.IMREAD_GRAYSCALE)

        # img = cv2.bitwise_and(img, img, mask=river_mask)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #masks = Image.open(mask_path)
        masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #mask = cv2.bitwise_and(mask, mask, mask=river_mask)
        #mask = cv2.resize(mask, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

        #masks = torch.as_tensor(mask, dtype=torch.int64)

        #mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        #mask = np.array(mask)
        # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        #masks = mask == obj_ids[:, None, None]
        #masks = torch.as_tensor(masks, dtype=torch.int64)

        # num_objs = len(obj_ids)

        # convert everything into a torch.Tensor
        #labels = torch.ones((num_objs,), dtype=torch.int64)

        # image_id = torch.tensor([idx])

        target = masks
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id

        img = self.transform(img)
        target = self.transform_target(target)

        target = np.array(target, dtype=np.int64)
        target = torch.as_tensor(target, dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)
        

def dataLoader():

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor()])

    transform_target = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224))])

    # use our dataset and defined transformations
    dataset = RiverDataset("floodwall_dataset_no_floodwall_v1.3", transform, transform_target)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=6,
        pin_memory=True)

    return train_loader

# Load dataset
TRAIN_LOADER = dataLoader()

pop_mean = []
pop_std0 = []
pop_std1 = []
for i, (image, label) in enumerate(TRAIN_LOADER, 0):
    if i == 0:
        print(image.shape)
    # shape (batch_size, 3, height, width)
    numpy_image = image.numpy()
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
pop_std1 = np.array(pop_std1).mean(axis=0)

print("normalize = transforms.Normalize(mean=[" + str(pop_mean[0]) + ", " + str(pop_mean[1]) + ", " + str(pop_mean[2]) + "], std=[" + str(pop_std0[0]) + ", " + str(pop_std0[1]) + ", " +  str(pop_std0[2]) + "])")
print("\n")
print("[" + str(pop_mean[0]) + ", " + str(pop_mean[1]) + ", " + str(pop_mean[2]) + "]")
print("[" + str(pop_std0[0]) + ", " + str(pop_std0[1]) + ", " + str(pop_std0[2]) + "]")
#print(pop_std0)
#print(pop_std1)
