# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import snntorch.functional as snnfunc

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import flif_snn
import random
import timeit


def main():

        # Network size and device setup
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        scale = 10
        num_input = scale*scale
        num_hidden1 = 1000
        num_hidden2 = 750
        num_output = 10

        num_steps = 200

        # Define network arch
        # This block you need to change
        #-----------------------------------------------------------------------------------------
        net = flif_snn.SNN(num_input, num_hidden1, num_output, num_steps, device).to(device)

        # Load learned weights into the network
        loader = np.load("../MNIST_Training/parameters.npz", allow_pickle = True)
        hid_con = loader['hid_con']
        out_con = loader['out_con']

        hid_con = torch.tensor(hid_con).to(device)
        out_con = torch.tensor(out_con).to(device)

        net.load(hid_con, out_con)
        
        #-----------------------------------------------------------------------------------------

        # Pre-created set of test batches (can be generated using spiked_data_builder.py)
        loader = np.load("../MNIST_Training/dataset.npz", allow_pickle = True)
        all_data = torch.tensor(loader['dat']).to(device)
        all_targets = torch.tensor(loader['tar']).to(device)
        print("Data loaded successfully")
        
        num_batches = all_targets.size(0)

        batch = random.randint(0, num_batches - 1)

        data = all_data[batch].clone()          
        targets = all_targets[batch]

        spikes, mem, _ = net(data, targets, False)

        in_spk = data.detach().cpu().numpy()
        hid_spk = net.hid_spk.detach().cpu().numpy()
        out_spk = spikes.detach().cpu().numpy()
        tar = targets.detach().cpu().numpy()

        print(snnfunc.acc.accuracy_rate(spikes, targets))

        cont = input("Continue? ")

        if cont != "y":
                quit()

        
        # Destination for results (may modify)
        np.savez("../MNIST_Training/smpl.npz", in_spk=in_spk, hid_spk=hid_spk, out_spk=out_spk, tar=tar)
        print("Saved data.")

                        
                

if __name__ == '__main__':
        main()
