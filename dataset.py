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
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        # """
        # :param image_paths: list of path to images
        # :param targets: numpy array
        # :param resize: tuple, e.g. (256, 256), resizes image if not None
        # :param augmentations: albumentation augmentations
        # """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.totensor = torchvision.transforms.ToTensor()
        # self.augmentations = augmentations

    def __len__(self):
        # """
        # Return the total number of samples in the dataset
        # """
        return len(self.image_paths)

    def demosaic_raw(self, meas):

        tform = skimage.transform.SimilarityTransform(rotation=0.00174)
        X = meas.numpy()[0, :, :]
        X = X / 65535.0
        X = X + 0.003 * np.random.randn(X.shape[0], X.shape[1])
        im1 = np.zeros((512, 640, 4))
        im1[:, :, 0] = X[0::2, 0::2]  # b
        im1[:, :, 1] = X[0::2, 1::2]  # gb
        im1[:, :, 2] = X[1::2, 0::2]  # gr
        im1[:, :, 3] = X[1::2, 1::2]  # r
        im1 = skimage.transform.warp(im1, tform)
        im = im1[6:506, 10:630, :]
        rowMeans = im.mean(axis=1, keepdims=True)
        colMeans = im.mean(axis=0, keepdims=True)
        allMean = rowMeans.mean()
        im = im - rowMeans - colMeans + allMean
        im = im.astype("float32")
        meas = torch.from_numpy(np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)).unsqueeze(0)

        # meas = np.swapaxes(
        #     np.swapaxes(im, 0, 2), 1, 2)
        # meas = torch.tensor(meas, dtype=torch.float32).unsqueeze(0)

        # meas = transforms.ToTensor()(np.array(meas))
        # return meas

        return meas[0, :, :, :]

    def __getitem__(self, item):
        # # """
        # # For a given "item" index, return everything we need
        # # to train a given model
        # # """
        # # use PIL to open the image
        # image = Image.open(self.image_paths[item])
        # # convert image to RGB, we have single channel images
        # image = image.convert("RGB")
        # # grab correct targets
        # targets = self.targets[item]
        # # resize if needed
        # if self.resize is not None:
        #     image = image.resize(
        #         (self.resize[1], self.resize[0]),
        #         resample=Image.BILINEAR
        #     )
        # # convert image to numpy array
        #     image = np.array(image)
        # # if we have albumentation augmentations
        # # add them to the image

        # meas image
        meas = Image.open(self.image_paths[item])
        meas = self.totensor(meas)
        meas = self.demosaic_raw(meas)

        # gt image
        gt_image = Image.open(self.targets[item])
        gt_image = gt_image.convert("RGB")
        gt_image = gt_image.resize(
            (self.resize[1], self.resize[0]), resample=Image.BILINEAR
        )
        gt_image = np.array(gt_image)
        gt_image = np.swapaxes(gt_image, 0, -1)

        return meas, gt_image
