# imports
import snntorch as snn
from snntorch import spikegen

import torch

import numpy as np
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Builds and stores a rate-encoded version of MNIST for robustness testing
def main():
        
        batch_size = 128
        data_path='/tmp/data/mnist'

        dtype = torch.float
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        scale = 10

        transform = transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

        test_batch = iter(test_loader)

        all_data = list()
        all_targets = list()

        num_steps = 200

        for data, targets in test_batch:

                data = data.to(device)
                targets = targets.to(device)
                targets = targets



                all_data.append(data.view(batch_size, - 1))
                all_targets.append(targets)


        all_data = torch.stack(all_data).detach().cpu().numpy()
        all_targets = torch.stack(all_targets).detach().cpu().numpy()

        np.savez("../MNIST_Training/rawdata.npz", dat=all_data, tar=all_targets)

        
if __name__ == '__main__':
        main()
