import numpy as np
import matplotlib.pyplot as plt
from cv2 import imshow
from sklearn.model_selection import train_test_split

meas_images = np.load("/home/thesis_bk/Emnist_dataset/emnist_measures.npy")
real_images = np.load("/home/thesis_bk/Emnist_dataset/emnist_imgs.npy")

# num_of_images = len(meas_images)
# print("number of images are", str(num_of_images))
# 112800
# num_of_images = len(real_images)
# print("number of images are", str(real_images))  # 112800

train_X, test_X, train_Y, test_Y = train_test_split(
    meas_images, real_images, test_size=0.10
)

train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.15)

print(len(train_X))
