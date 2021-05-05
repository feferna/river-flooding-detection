import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

import torch
from torchvision import transforms, models


def load_initial_model(backbone_name, dataset_name, model_file):
    if backbone_name == "ResNet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, aux_loss=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, aux_loss=True)
    
    if dataset_name == "SaoCarlos":
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    elif dataset_name == "PennFudanPed":
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

        checkpoint = torch.load(model_file)

        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_trained_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    model.cpu()
    
    checkpoint = torch.load("./SaoCarlos_trained_model_ResNet50.pth",  map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()

    return model

def main():
    # ####################################################
    # Test model in a single image and show the results
    # ####################################################
    model = load_trained_model()   
    
    # Test img is the latest_img
    test_img = cv2.imread("./IMG_TEST6.jpg")
    
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
    normalize = transforms.Normalize(mean=[0.5291628, 0.5135074, 0.45702627], std=[0.19916107, 0.18686345, 0.1919754])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        normalize])

    test_img_tensor = transform(test_img)
    test_img_tensor = test_img_tensor[None, :, :]

    output = model(test_img_tensor)["out"]
    _, preds = torch.max(output, 1)
    preds = preds[0, :, :].detach().cpu().numpy()

    obj_ids = np.unique(preds)
    masks = preds == obj_ids[:, None, None]

    segmentation_rgb = np.zeros(shape=(masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
    segmentation_rgb[:, :, 0] = masks[1] * 255

    river_level_img = segmentation_rgb[:, :, 0]

    river_level_img = cv2.resize(river_level_img, (720,720), cv2.INTER_NEAREST)

    segmentation_img = np.zeros(shape=(river_level_img.shape[0], river_level_img.shape[1], 3), dtype=np.uint8)
    segmentation_img[:, :, 0] = river_level_img

    seg_img = np.zeros(shape=test_img.shape, dtype=np.uint8)

    # compute center offset
    x = (test_img.shape[1] - 720) // 2
    y = (test_img.shape[0] - 720) // 2

    seg_img[y: (y + 720), x: (x + 720), :] = segmentation_img
    
    test_img = cv2.addWeighted(test_img[y: (y + 720), x: (x + 720), :], 0.8, segmentation_img, 0.2, 0)

    plt.imshow(test_img)
    plt.show()

    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)    
    cv2.imwrite("./test_img.jpg", test_img)



if __name__ == "__main__":
    main()
