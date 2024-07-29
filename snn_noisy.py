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
        num_hidden2 = 750
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

        # Pre-created set of test batches (can be generated using data_builder.py)
        # NOTE: this data needs to be raw (unspiked), so that we can perturb
        # the grayscale values and THEN generate spikes.
        loader = np.load("../MNIST_Training/rawdata.npz", allow_pickle = True)
        all_data = torch.tensor(loader['dat']).to(device)
        all_targets = torch.tensor(loader['tar']).to(device)
        print("Data loaded successfully")
        
        num_batches = all_targets.size(0)
        batch_size = all_targets.size(1)


        num_samples = 100
        num_perturbed = 100

        accuracy = np.zeros(num_samples)

        num_batches = 20
        

        print("Beginning testing")
        for i in range(0, num_perturbed+1, 10):

                for j in range(num_samples):

                        start = timeit.default_timer()
                        
                        # Select i inputs to add noise to
                        neurons = random.sample(range(0, 100), i)
                                
                        # Generate noise for the inputs
                        # num_batches x batch_size x num_input
                        noise = np.random.rand(num_batches, batch_size, i)
                        
                        acc = 0
                        for batch in range(num_batches):

                                data = all_data[batch].clone() # batch_size x num_input

                                
                                # Then spike

                                spiked_data = list()
                                
                                for sample in data:

                                        spiked_sample = spikegen.rate(sample, num_steps = num_steps)
                                        spiked_data.append(spiked_sample)

                                spiked_data = torch.stack(spiked_data)

                                # Add noise
                                for k in range(i):

                                        data[:, :, neurons[k]] = torch.tensor(noise[batch, :, k]).to(device)
                                
                                targets = all_targets[batch]

                                # Change this line; function call likely won't work for your net
                                spikes, _, _ = net(spiked_data, targets, False)

                                acc += snnfunc.acc.accuracy_rate(spikes, targets)

                        acc = acc / num_batches
                        
                        accuracy[j] = acc

                        end = timeit.default_timer()
                        print(f"Sample {j} completed for {i} noisy inputs. Time elapsed: {end-start}")

                np.savez("../MNIST_Training/" + model + "_noisy/mc_noisy_" + model + "_" + str(i) + ".npz", acc=accuracy)
                print(f"Saved data for {i} perturbations")
        
        # Destination for results (may modify)
        

                        
                

if __name__ == '__main__':
        main("ce")
        main("mse")
