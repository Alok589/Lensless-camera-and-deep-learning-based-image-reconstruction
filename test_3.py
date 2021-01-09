import os
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import dataset
import torch.optim as optim
from torch.optim import lr_scheduler, optimizer
from Dense_Unet import Dense_Unet
from PIL import Image
import cv2
import engine
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.hub import tqdm

# from torch.nn.parallel import DistributedDataParallel as DDP
from skimage.filters import rank
import scipy.io as sio


if __name__ == "__main__":

    project_path = "/home/thesis_2/"
    data_path = "/home/thesis_2/Emnist_dataset/"
    
    loss_curves = os.path.join(project_path, "loss_curves")
    models_weights = os.path.join(project_path, "models_weights")
    optimizer_chp = os.path.join(project_path, "optimizer_chp")

    file_names = ["emnist_imgs.npy", "emnist_measures.npy", "emnist_labels.npy"]
      


    exp = "exp_24_MAE"
    device = "cuda:6"
    epochs = 50
    is_model_trained = False
    ck_pt_path = "/home/thesis_bk/optimizer_chp/exp_20.pt"



    if is_model_trained:
        checkpoint = torch.load(ck_pt_path)

    meas_images = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")
    real_images = np.load("/home/thesis_2/Emnist_dataset/emnist_imgs.npy")
    labels = np.load("/home/thesis_2/Emnist_dataset/emnist_labels.npy")

    # num_of_images = len(meas_images)
    # print("number of images are", str(num_of_images))
    # 112800
    # num_of_images = len(real_images)
    # print("number of images are", str(real_images))  # 112800

    img_indices = np.arange(112800)

    train_indices, test_indices , _, _ = train_test_split(img_indices, img_indices, test_size= 0.10)

    train_indices, val_indices, _, _ = train_test_split(train_indices, train_indices, test_size= 0.10)

    # indices are done 
    train_X = meas_images[train_indices]
    val_X = meas_images[val_indices]
    test_X = meas_images[test_indices]

    train_Y = real_images[train_indices]
    val_Y = real_images[val_indices]
    test_Y = real_images[test_indices]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    val_labels = labels[val_indices]








    # train_X, test_X, train_Y, test_Y = train_test_split(
    #     meas_images, real_images, test_size=0.10
    # )

    # train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.15)

    model = Dense_Unet()

    if is_model_trained:
        model.load_state_dict(checkpoint["model"])

    # move model to device
    model.to(device)

    train_dataset = dataset.ClassificationDataset(
        meas_images = train_X, real_images = train_Y, labels=train_labels


    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2
    )


    valid_dataset = dataset.ClassificationDataset(
        meas_images = val_X, real_images = val_Y, labels=val_labels
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=2
    )

    test_dataset = dataset.ClassificationDataset(
        meas_images = test_X, real_images = test_Y, labels=test_labels
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )


    # i = 0 
    # for meas, real, label in train_loader:
    #     m = meas 
    #     r = real
    #     l = label;
    #     if i == 1:
    #         break
    #     i+=1

    print(" ")
            