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

import flif_snn
import random
import timeit


def main(model):

        
        # Network size and device setup
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        scale = 10
        num_input = scale*scale
        num_hidden1 = 1000
        num_output = 10

        num_steps = 200

        # Define network arch
        # This block you need to change
        #-----------------------------------------------------------------------------------------
        net = flif_snn.SNN(num_input, num_hidden1, num_output, num_steps, device).to(device)

        # Load learned weights into the network
        loader = np.load("../MNIST_Training/parameters_" + model + ".npz", allow_pickle = True)
        hid_con = loader['hid_con']
        out_con = loader['out_con']

        hid_con = torch.tensor(hid_con).to(device)
        out_con = torch.tensor(out_con).to(device)

        success = net.load(hid_con, out_con)
        
        if success == -1:
                quit()
        #-----------------------------------------------------------------------------------------

        # Pre-created set of test batches (can be generated using spiked_data_builder.py)
        loader = np.load("../MNIST_Training/dataset.npz", allow_pickle = True)
        all_data = torch.tensor(loader['dat']).to(device)
        all_targets = torch.tensor(loader['tar']).to(device)
        print("Data loaded successfully")
        
        num_batches = all_targets.size(0)
        batch_size = all_targets.size(1)

        num_samples = 100
        num_deleted = 100

        num_batches = 20

        accuracy = np.zeros(num_samples)

        print("Beginning testing")
        for i in range(0, num_deleted+1, 10):

                for j in range(num_samples):

                        start = timeit.default_timer()
                        
                        deleted = list()
                        
                        deleted = random.sample(range(0, 100), i)

                        
                        acc = 0
                        for batch in range(num_batches):

                                data = all_data[batch].clone()
                                
                                for neuron in deleted:

                                        data[:, :, neuron] = torch.zeros_like(data[:, :, neuron])
                                
                                targets = all_targets[batch]

                                # Change this line; function call likely won't work for your net
                                spikes, _, _ = net(data, targets, False)

                                no_output = spikes.sum().item()

                                if no_output == 0:
                                        acc += 0
                                else:
                                        acc += snnfunc.acc.accuracy_rate(spikes, targets)

                        acc = acc / num_batches
                        
                        accuracy[j] = acc

                        end = timeit.default_timer()
                        print(f"Sample {j} completed for {i} deletions. Time elapsed: {end-start}")


                np.savez("../MNIST_Training/" + model + "_dead/mc_dead_" + model + "_" + str(i) + ".npz", acc=accuracy)
                print(f"Saved data for {i} deletions.")
        
        # Destination for results (may modify)
        
        
                        
                

if __name__ == '__main__':
        main("mse")
        main("ce")
