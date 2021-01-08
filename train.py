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

    project_path = "/home/thesis_bk/"
    data_path = "/home/thesis_bk/dataset/"
    file_name = "file_name.txt"
    loss_curves = os.path.join(project_path, "loss_curves")
    models_weights = os.path.join(project_path, "models_weights")
    optimizer_chp = os.path.join(project_path, "optimizer_chp")

    exp = "exp_24_MAE"
    device = "cuda:6"
    epochs = 50
    is_model_trained = False
    ck_pt_path = "/home/thesis_bk/optimizer_chp/exp_20.pt"

    if is_model_trained:
        checkpoint = torch.load(ck_pt_path)

    with open(file_name, "r") as f:
        file_names = f.read()[:-1].split("\n")

    meas_data = [
        os.path.join(data_path, "measurements", file.split("_")[0], file + "..png")
        for file in file_names
    ]
    gt_data = [
        os.path.join(data_path, "groundtruth", file.split("_")[0], file + ".JPEG")
        for file in file_names
    ]

    train_X, test_X, train_Y, test_Y = train_test_split(
        meas_data, gt_data, test_size=0.20
    )

    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.15)

    # # binary targets numpy array
    # targets = df.target.values
    # # fetch out model, we will try both pretrained
    # # and non-pretrained weights
    model = Dense_Unet()

    if is_model_trained:
        model.load_state_dict(checkpoint["model"])

    # move model to device
    model.to(device)

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_X, targets=train_Y, resize=(256, 256)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2
    )

    valid_dataset = dataset.ClassificationDataset(
        image_paths=val_X, targets=val_Y, resize=(256, 256)
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=2
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_X, targets=test_Y, resize=(256, 256)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    # init = "Transpose"

    # if init == "Transpose":
    #     print("Loading calibrated files")
    #     d = sio.loadmat("data/flatcam_prototype2_calibdata.mat")
    #     phil = np.zeros((500, 256, 1))
    #     phir = np.zeros((620, 256, 1))
    #     phil[:, :, 0] = d["P1gb"]
    #     phir[:, :, 0] = d["Q1gb"]
    #     phil = phil.astype("float32")
    #     phir = phir.astype("float32")
    # else:
    #     print("Loading Random Toeplitz")
    #     phil = np.zeros((500, 256, 1))
    #     phir = np.zeros((620, 256, 1))
    #     pl = sio.loadmat("data/phil_toep_slope22.mat")
    #     pr = sio.loadmat("data/phir_toep_slope22.mat")
    #     phil[:, :, 0] = pl["phil"][:, :, 0]
    #     phir[:, :, 0] = pr["phir"][:, :, 0]
    #     phil = phil.astype("float32")
    #     phir = phir.astype("float32")

    writer = SummaryWriter("tensorboard/" + exp + "/")

    # tb = SummaryWriter()
    # model = Dense_Unet()
    # images_batch = torch.from_numpy(np.array(train_X, dtype='int32'))
    # images = next(iter(images_batch))

    # grid = torchvision.utils.make_grid(images)
    # tb.add_images('images', grid)
    # tb.add_graph(model, images)
    # tb.close()
    # data = train_dataset[23]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if is_model_trained:
        # optimizer = optimizer.load.checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint["optimizer"])

    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=3, verbose=True, factor=0.2)
    # train and print auc score for all epochs^

    if is_model_trained:
        start_epoch = checkpoint["epoch"]
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

        if epoch % 25 == 0:
            print(epoch)
    writer.close()

    # torch.save(model.state_dict(), os.path.join(models_weights, exp + ".pt"))

    checkpoint = {
        "epoch": epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # torch.save(checkpoint, os.path.join(optimizer_chp, exp + ".pt"))
    torch.save(checkpoint, os.path.join(optimizer_chp, "exp_24_MAE.pt"))
    # checkpoint = torch.load('checkpoint.pth')

    plt.figure()
    plt.plot(list(range(1, epochs + 1)), train_losses, label="train")
    plt.plot(list(range(1, epochs + 1)), val_losses, label="val")
    plt.legend()

    plt.savefig(os.path.join(loss_curves, exp + ".png"))

    print("")

    # wrap model and optimizer with NVIDIA's apex
    # this is used for mixed precision training
    # if you have a GPU that supports mixed precision,
    # this is very helpful as it will allow us to fit larger images
    # and larger batches
#     model, optimizer=amp.initialize(
#     model, optimizer, opt_level="O1", verbosity=0
#     )
# if we have more than one GPU, we can use both of them!
# if torch.cuda.device_count() > 1:

#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model, device_ids=[1, 2, 3])
# some logging
# print(f"Training batch size: {TRAINING_BATCH_SIZE}")
# print(f"Test batch size: {TEST_BATCH_SIZE}")
# print(f"Epochs: {EPOCHS}")
# print(f"Image size: {IMAGE_SIZE}")
# print(f"Number of training images: {len(train_dataset)}")
# print(f"Number of validation images: {len(valid_dataset)}")
# print(f"Encoder: {ENCODER}")

# # loop over all epochs
# for epoch in range(EPOCHS):
#     print(f"Training Epoch: {epoch}")
#     # train for one epoch
#     train(
#     train_dataset,
#     train_loader,
#     model,
#     criterion,
#     optimizer
#     )

# print(f"Validation Epoch: {epoch}")
# # calculate validation loss
# val_log=evaluate(
# valid_dataset,
# valid_loader,
# model
# )
# # step the scheduler
# scheduler.step(val_log["loss"])
# print("\n")
