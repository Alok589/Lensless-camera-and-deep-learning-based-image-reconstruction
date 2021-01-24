# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm

# import model
from torch._C import device
from torch.optim import optimizer
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from Dense_Unet import Dense_Unet
from torchvision.io import image
from piqa import ssim
import pytorch_ssim


from piqa import ssim


class SSIMLoss(ssim.SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


crit = SSIMLoss()


def train(data_loader, model, optimizer, device):

    model.train()
    batch_MSEs = []
    for data in tqdm(data_loader):
        # remember, we have image and targets
        # in our dataset class
        inputs = data[0]
        targets = data[1]
        labels = data[2]
        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs)

        "BCE_LOSS"
        loss = torch.nn.BCELoss()(outputs, targets)

        "MSE_LOSS"
        # loss = nn.MSELoss()(outputs, targets)

        "SSIM_LOSS"
        # criterion = 1 - pytorch_ssim.ssim()(outputs, targets)
        # loss = criterion()
        # crit = SSIMLoss().cuda(device)
        # ssim = crit(outputs, targets)
        # loss = ssim

        "MAE_LOSS"
        # loss = torch.abs(targets - outputs).mean()

        batch_MSEs.append(loss.item())

        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # print(loss.item())
    batch_MSEs = np.array(batch_MSEs)
    epoch_loss = np.mean(batch_MSEs)
    print(epoch_loss)
    return epoch_loss


def evaluate(data_loader, model, device):

    # """
    # This function does evaluation for one epoch
    # :param data_loader: this is the pytorch dataloader
    # :param model: pytorch model
    # :param device: cuda/cpu
    # """

    # put model in evaluation mode
    print("_____________validation_____________")
    model.eval()

    # init lists to store targets and outputs
    batch_MSEs = []
    # # we use no_grad context
    with torch.no_grad():
        for idx, data in enumerate(data_loader, 1):
            inputs = data[0]
            targets = data[1]
            labels = data[2]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            # do the forward step to generate prediction
            model.to(device)
            outputs = model(inputs)

            "SSIM_LOSS"
            # crit = SSIMLoss().cuda(device)
            # ssim = crit(outputs, targets)
            # loss = ssim

            "MSE_LOSS"
            # batch_mse = ((outputs - targets) ** 2).mean().item()
            #         #print("batch"+str(idx) + " loss:" ,batch_mse)

            "MAE_loss"
            # batch_mse = torch.abs(output - targets).mean().item()

            "BCE_LOSS"
            batch_mse = torch.nn.BCELoss()(outputs, targets).item()
            # batch_mse = 1 - pytorch_ssim.ssim(img1, img2, windoe_size=11).item()
            batch_MSEs.append(batch_mse)
        #         # return final output and final targets
        batch_MSEs = np.array(batch_MSEs)
        epoch_loss = np.mean(batch_MSEs)
        print(epoch_loss)
    return epoch_loss

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            targets = data[1]
            labels = data[2]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            output = model(inputs)
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            final_targets.extend(targets)
            final_outputs.extend(output)
    return final_outputs, final_targets
