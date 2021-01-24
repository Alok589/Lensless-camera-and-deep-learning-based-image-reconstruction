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
from Deep_Res_Unet import Deep_Res_Unet
from engine import evaluate
import cv2

# writer = SummaryWriter("runs/thesis")


tform = tf.SimilarityTransform(rotation=0.00174)


# Specify the path to the measurement


# plot_name = "exp_15_epoc_80_" + str(1) + ".png"
# plot_name = "check_1" + ".png"
model = Deep_Res_Unet()
model.load_state_dict(torch.load("/home/thesis_2/model_opt_chp/exp_04.pt")["model"])
model.eval()
device = "cuda:5"

# X = skimage.io.imread(
#     "/home/thesis_bk/dataset/measurements/n01440764/n01440764_457..png"
# )

image_idx = 1000
x = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]
y = np.load("/home/thesis_2/Emnist_dataset/emnist_imgs.npy")[image_idx]
y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
z = np.load("/home/thesis_2/Emnist_dataset/emnist_measures.npy")[image_idx]


x = np.expand_dims(x, 0)
x = np.expand_dims(x, 0)
data = torch.tensor(x)
op = model(data)

pred = op[0].detach().numpy()[0]

# print("recon", recn)
# print(recn.shape)
# recn = np.swapaxes(recn, 0, 1)

# plt.figure()
# skimage.io.imshow(op)
# io.show()
# plt.savefig(os.path.join("/home/thesis_2/inference_plots", plot_name))


fig = plt.figure()

plt.subplot(1, 4, 1)
plt.imshow(z)
plt.title("measurements")


plt.subplot(1, 4, 2)
plt.imshow(pred)
plt.title("prediction")

plt.subplot(1, 4, 3)
plt.imshow(y)
plt.title("Real_image")

plt.subplot(1, 4, 4)
plt.imshow(np.abs(pred - y))
plt.title("|pred - Real_image|")
plot_name = "inf_04" + ".png"
# plt.savefig("infe_3.png")
plt.savefig(os.path.join("/home/thesis_2/inference_plots", plot_name))


##############
# plt.figure()
# plt.imshow(pred)
# plt.savefig("inference_1.png")


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