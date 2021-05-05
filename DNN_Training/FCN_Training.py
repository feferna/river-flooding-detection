import os

import torch
from torchvision import transforms, models

import numpy as np

import cv2

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score as IoU

from DataLoader import data_loader


def main():
    dataset = "SaoCarlos"
    # dataset = "PennFudanPed"
    #dataset = "PascalVOC"
    
    # backbone = "ResNet50"
    backbone = "ResNet101"

    epochs = 50
    batch_size = 8

    ##################################
    train_loader, test_loader = data_loader(dataset, batch_size)

    if backbone == "ResNet50":
        model = models.segmentation.fcn_resnet50(pretrained=True)
    else:
        model = models.segmentation.fcn_resnet101(pretrained=True)
    
    if dataset == "PascalVOC":
        model.classifier[4] = torch.nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001,
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    elif dataset == "SaoCarlos":
        model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001,
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.5)

    else: # PennFudan
        model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.cuda().train()

    criterion = torch.nn.CrossEntropyLoss()

    # ################
    # Train model
    # ################
    loss_arr = []
    mean_IoU_arr = []

    for epoch in range(epochs+1):
        print("Epoch: " + str(epoch))
        running_loss = 0

        if dataset == "PascalVOC":
            IoU_accumulated = [0.0]*21
        else:
            IoU_accumulated = [0.0, 0.0]

        num_samples = 0
        batch_counter = 0
        for images, labels in train_loader:
            batch_counter += 1
            images = images.cuda()

            if dataset == "PascalVOC":
                labels = labels.squeeze()
                labels = labels*255
                labels[np.where(labels == 255)] = 0
                labels = labels.long()

            labels = labels.cuda()

            num_samples += len(labels)

            model_outputs = model(images)['out']

            _, preds = torch.max(model_outputs, dim=1)

            batch_loss = criterion(model_outputs, labels)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

            ground_truth = labels.cpu().numpy().reshape(-1)
            predicted = preds.cpu().numpy().reshape(-1)

            if dataset == "PascalVOC":
                list_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                IoU_accumulated += IoU(predicted, ground_truth, labels=list_labels, average=None, zero_division=0.0)
            else:
                IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)


        epoch_loss = running_loss / num_samples
        loss_arr.append(epoch_loss)

        IoU_accumulated = IoU_accumulated / batch_counter
        print(IoU_accumulated)
        mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)

        mean_IoU_arr.append(mean_IoU)

        lr_scheduler.step()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': epoch_loss,
            }, "./" + dataset + "_trained_model_" + backbone + ".pth")

        print("\tMean Loss: {:.4f}".format(epoch_loss))
        print("\tMean IoU: {:.4f}".format(mean_IoU))

    # ##########################
    # Plot training statistics
    # ##########################
    fig, host = plt.subplots(nrows=1, ncols=1)

    par1 = host.twinx()

    host.set_xlabel("Epochs")
    host.set_ylabel("Loss")
    par1.set_ylabel("Intersection over Union (IoU)")

    p1, = host.plot(loss_arr, color='r', linestyle='--', marker='o',  label="Training Loss")

    if dataset == "SaoCarlos":
        p2, = par1.plot(mean_IoU_arr, color='b', linestyle='--', marker='*', label="Training IoU")
    else:
        p2, = par1.plot(mean_IoU_arr, color='b', linestyle='--', marker='*', label="Training IoU")
  

    lns = [p1, p2]
    host.legend(handles=lns, loc='center right')
    host.set_title("FCN with " + backbone +  " backbone")

    plt.savefig(dataset + "training_FCN_" + backbone + ".pdf", bbox_inches='tight')
    

    # ############################
    # Test model in the test set
    # ############################
    IoU_accumulated = [0.0, 0.0]
    running_loss = 0
    batch_counter = 0
    num_samples = 0
    for images, labels in test_loader:
        batch_counter += 1
        images = images.cuda()

        if dataset == "PascalVOC":
            labels = labels.squeeze()
            labels = labels*255
            labels[np.where(labels == 255)] = 0
            labels = labels.long()

        labels = labels.cuda()

        num_samples += len(labels)

        model_outputs = model(images)['out']

        _, preds = torch.max(model_outputs, 1)

        batch_loss = criterion(model_outputs, labels)

        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()

        ground_truth = labels.cpu().numpy().reshape(-1)
        predicted = preds.cpu().numpy().reshape(-1)

        if dataset == "PascalVOC":
            list_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            IoU_accumulated += IoU(predicted, ground_truth, labels=list_labels, average=None, zero_division=0.0)
        else:
            IoU_accumulated += IoU(predicted, ground_truth, labels=[0, 1], average=None)
    
    epoch_loss = running_loss / num_samples

    IoU_accumulated = IoU_accumulated / batch_counter
    mean_IoU = np.sum(IoU_accumulated[1:]) / (len(IoU_accumulated) - 1)

    print("\tTest Loss: {:.4f}".format(epoch_loss))
    print("\tTest IoU: {:.4f}".format(mean_IoU))


if __name__ == "__main__":
    main()
