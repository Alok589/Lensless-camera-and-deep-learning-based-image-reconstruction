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
from skimage.filters import rank
import scipy.io as sio
import skimage.transform as skt
from PIL import Image
from PIL import ImageFile
from Deep_Res_Unet import Deep_Res_Unet


if __name__ == "__main__":

    project_path = "/home/thesis_2/"
    data_path = "/home/thesis_2/Emnist_dataset/"

    loss_curves = os.path.join(project_path, "loss_curves")
    # models_weights = os.path.join(project_path, "models_weights")
    model_opt_chp = os.path.join(project_path, "model_opt_chp")

    file_names = ["emnist_imgs.npy", "emnist_measures.npy", "emnist_labels.npy"]

    exp = "exp_05"
    device = "cuda:5"
    # device = torch.device("cuda:0")
    epochs = 50
    is_model_trained = False
    ck_pt_path = "/home/thesis_2/model_opt_chp/exp_20.pt"

    if is_model_trained:
        checkpoint = torch.load(ck_pt_path)

    meas_images = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")
    real_images = np.load("/home/thesis_2/Emnist_dataset/emnist_imgs.npy")
    labels = np.load("/home/thesis_2/Emnist_dataset/emnist_labels.npy")

    img_indices = np.arange(112800)

    train_indices, test_indices, _, _ = train_test_split(
        img_indices, img_indices, test_size=0.20
    )

    train_indices, val_indices, _, _ = train_test_split(
        train_indices, train_indices, test_size=0.20
    )

    # indices are done
    train_X = meas_images[train_indices][:70000]
    val_X = meas_images[val_indices][:70000]
    test_X = meas_images[test_indices][:70000]

    train_Y = real_images[train_indices][:70000]
    val_Y = real_images[val_indices][:70000]
    test_Y = real_images[test_indices][:70000]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    val_labels = labels[val_indices]

    model = Deep_Res_Unet()

    if is_model_trained:
        model.load_state_dict(checkpoint["model"])

    # move model to device
    model.to(device)

    train_dataset = dataset.ClassificationDataset(
        meas_images=train_X, real_images=train_Y, labels=train_labels
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2
    )

    valid_dataset = dataset.ClassificationDataset(
        meas_images=val_X, real_images=val_Y, labels=val_labels
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=2
    )

    test_dataset = dataset.ClassificationDataset(
        meas_images=test_X, real_images=test_Y, labels=test_labels
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    writer = SummaryWriter("tensorboard/" + exp + "/")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if is_model_trained:
        # optimizer = optimizer.load.checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint["optimizer"])

    if is_model_trained:
        tart_epoch = checkpoint["epoch"]
        end_epoch = checkpoint["epoch"] + epochs
    else:
        start_epoch = 0
        end_epoch = epochs

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(start_epoch, end_epoch)):
        print("epoch " + str(epoch))
        train_loss = engine.train(train_loader, model, optimizer, device=device)
        val_loss = engine.evaluate(valid_loader, model, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("train", train_loss, epoch)
        writer.add_scalar("val", val_loss, epoch)
        writer.add_scalars(
            "train and val losses", {"train": train_loss, "val": val_loss}, epoch
        )

        # if epoch % 25 == 0:
        #     print(epoch)
    writer.close()

    # torch.save(model.state_dict(), os.path.join(models_weights, exp + ".pt"))

    checkpoint = {
        "epoch": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # # torch.save(checkpoint, os.path.join(optimizer_chp, exp + ".pt"))
    torch.save(checkpoint, os.path.join(model_opt_chp, "exp_05.pt"))
    # # checkpoint = torch.load('checkpoint.pth')

    plt.figure()
    plt.plot(list(range(1, epochs + 1)), train_losses, label="train")
    plt.plot(list(range(1, epochs + 1)), val_losses, label="val")
    plt.legend()

    plt.savefig(os.path.join(loss_curves, exp + ".png"))

    print("")