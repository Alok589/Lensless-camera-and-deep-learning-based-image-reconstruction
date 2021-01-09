import torch
import numpy as np
import torchvision
from PIL import Image
from PIL import ImageFile
import skimage
import torchvision.transforms as transforms

# sometimes, you will have images without an ending bit
# this takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, meas_images, real_images, labels):
        self.meas_images = meas_images 
        self.real_images = real_images 
        self.labels = labels     

    def __len__(self):
        return len(self.meas_images)

    def __getitem__(self, item):
    # meas = Image.open(self.image_paths[item])
    # meas = self.totensor(meas)
    # meas = self.demosaic_raw(meas)

    # # gt image
    # gt_image = Image.open(self.targets[item])
    # gt_image = gt_image.convert("RGB")
    # gt_image = gt_image.resize(
    #     (self.resize[1], self.resize[0]), resample=Image.BILINEAR
    # )
    # gt_image = np.array(gt_image)
    # gt_image = np.swapaxes(gt_image, 0, -1)

        meas = self.meas_images[item]
        gt_image = self.real_images[item]
        labels = self.labels[item]        

        return meas, gt_image, labels



    # def demosaic_raw(self, meas):

    #     tform = skimage.transform.SimilarityTransform(rotation=0.00174)
    #     X = meas.numpy()[0, :, :]
    #     X = X / 65535.0
    #     X = X + 0.003 * np.random.randn(X.shape[0], X.shape[1])
    #     im1 = np.zeros((512, 640, 4))
    #     im1[:, :, 0] = X[0::2, 0::2]  # b
    #     im1[:, :, 1] = X[0::2, 1::2]  # gb
    #     im1[:, :, 2] = X[1::2, 0::2]  # gr
    #     im1[:, :, 3] = X[1::2, 1::2]  # r
    #     im1 = skimage.transform.warp(im1, tform)
    #     im = im1[6:506, 10:630, :]
    #     rowMeans = im.mean(axis=1, keepdims=True)
    #     colMeans = im.mean(axis=0, keepdims=True)
    #     allMean = rowMeans.mean()
    #     im = im - rowMeans - colMeans + allMean
    #     im = im.astype("float32")
    #     meas = torch.from_numpy(np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)).unsqueeze(0)
    #     return meas[0, :, :, :]

