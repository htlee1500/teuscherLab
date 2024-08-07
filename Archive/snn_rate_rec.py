# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import snntorch.functional as snnfunc

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import math
import timeit

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import Fractional_LIF
import flif_snn_rec


# MAIN FUNCTION
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

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        print(mnist_train)
        
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


        num_input = scale*scale
        num_hidden1 = 500
        num_hidden2 = 750
        num_output = 10

        num_steps = 200

        
        #net = flif_snn.SNN2(num_input, num_hidden1, num_hidden2, num_output, num_steps).to(device)
        net = flif_snn_rec.SNN(num_input, num_hidden1, num_output, num_steps, device).to(device)
        
        loss = snnfunc.ce_rate_loss()
        #loss = snnfunc.mse_count_loss(correct_rate = 0.8, incorrect_rate = 0.2)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas = (0.9, 0.999))


        num_epochs = 1
        counter = 0

        training_acc = list()
        testing_acc = list()

        torch.no_grad()
        
        for epoch in range(num_epochs):
                iter_counter = 0
                train_batch = iter(train_loader)

                plot = False

                
                for data, targets in train_batch:


                        
                        data = data.to(device)
                        targets = targets.to(device)
                        targets = targets
                        
                        net.train()

                        spiked_data = list()
                        for sample in data.view(batch_size, -1):

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)

                        spikes, mem  = net(spiked_data, targets, False)


                        net.weight_update()
                        

                        acc = snnfunc.acc.accuracy_rate(spikes, targets)
                        training_acc.append(acc)


                        with torch.no_grad():

                                net.eval()
                                test_data, test_targets = next(iter(test_loader))
                                test_data = test_data.to(device)
                                test_targets = test_targets.to(device)
                                test_targets = test_targets

                                spiked_test_data = list()
                                for sample in test_data.view(batch_size, -1):

                                        spiked_sample = spikegen.rate(sample, num_steps = num_steps)

                                        spiked_test_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                                spiked_test_data = torch.stack(spiked_test_data).to(device)

                                
                                test_spike, test_memory = net(spiked_test_data, targets, False)


                                test_acc = snnfunc.acc.accuracy_rate(test_spike, test_targets)
                                testing_acc.append(test_acc)
                                


                                if counter % 50 == 0:
                                        print(f"Iteration {iter_counter}. \n Accuracy at training: {acc} \n Accuracy at testing: {test_acc}")
                                        

                                counter += 1
                                iter_counter += 1


        print(f"Iterations: {iter_counter}")
        
        # get overall results for file
        print("Testing time:")
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle = True, drop_last = False)

        total = 0
        correct = 0
        with torch.no_grad():
                net.eval()
                for data, targets in test_loader:
                        data = data.to(device)
                        targets = targets.to(device)
                        targets = targets

                        spiked_data = list()
                        for sample in data.view(data.size(0), -1):

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)


                        test_spikes, _ = net(spiked_data, targets, False)

                        _, predicted = test_spikes.sum(dim=0).max(1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

        
        
        performance = 100 * correct / total
        print("Performance:", performance)

        net(spiked_data, targets, True)

        
                                
if __name__ == '__main__':
        main()
