from skimage import transform as tf
import torchvision
from skimage import transform
from Dense_Unet import Dense_Unet
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from cv2 import exp, transform
from numpy.lib.npyio import save
import dataset
import matplotlib.pyplot as plt


import torch
import scipy.io as sio
import numpy as np
import os
import skimage.io
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer

# writer = SummaryWriter("runs/thesis")


tform = tf.SimilarityTransform(rotation=0.00174)


def evaluate(X):
    X = X / 65535.0
    X_train = np.zeros((1, 4, 500, 620))
    im1 = np.zeros((512, 640, 4))
    im1[:, :, 0] = X[0::2, 0::2]  # b
    im1[:, :, 1] = X[0::2, 1::2]  # gb
    im1[:, :, 2] = X[1::2, 0::2]  # gr
    im1[:, :, 3] = X[1::2, 1::2]  # r
    im1 = tf.warp(im1, tform)
    im = im1[6:506, 10:630, :]
    rowMeans = im.mean(axis=1, keepdims=True)
    colMeans = im.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    im = im - rowMeans - colMeans + allMean

    X_train[0, :, :, :] = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
    X_train = X_train.astype("float32")
    X_val = torch.from_numpy(X_train)
    Xvalout = model(X_val)
    ims = Xvalout.detach().numpy()
    ims = np.swapaxes(np.swapaxes(ims[0, :, :, :], 0, 2), 0, 1)
    ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
    return ims

    # Specify the path to the measurement


# plot_name = "exp_15_epoc_80_" + str(1) + ".png"
plot_name = "check_5" + ".png"
model = Dense_Unet()
model.load_state_dict(
    torch.load("/home/thesis_bk/optimizer_chp/exp_24_MAE.pt")["model"]
)
model.eval()

# directory = "optimizer_chp"
# parent_dir = "/thesis/"
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)

# optim.load_state_dict(torch.load("/thesis/optimizer_chp/opt_1.pt"))


X = skimage.io.imread(
    "/home/thesis_bk/dataset/measurements/n01440764/n01440764_457..png"
)
recn = evaluate(X)
# print("recon", recn)
# print(recn.shape)
recn = np.swapaxes(recn, 0, 1)

plt.figure()
skimage.io.imshow(recn)
io.show()
plt.savefig(os.path.join("/home/thesis_bk/inference_plots", plot_name))


# directory = "models_weights"
# parent_dir = "/thesis/"
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)


# torch.save(model.state_dict(), path)
# model.load_state_dict(torch.load(path))
# model.eval(path="entire_model.pt")

# PATH = "entire_model.pt"

# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval(PATH="entire_model.pt")
