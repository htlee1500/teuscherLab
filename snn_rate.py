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
import flif_snn



def train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter):

        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print("\n")

def spiker(data, num_steps):

        #return spikegen.latency(data, num_steps=num_steps, tau=1, normalize = False)
        return spikegen.rate(data, num_steps=num_steps)
        
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
        num_hidden = 1000
        num_output = 10

        num_steps = 200

        
        net = flif_snn.SNN(num_input, num_hidden, num_output, num_steps, device).to(device)

        
        loss = snnfunc.ce_rate_loss()
        #loss = snnfunc.mse_count_loss()
        #loss = snnfunc.ce_temporal_loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas = (0.9, 0.999))


        num_epochs = 5
        loss_hist = list()
        test_loss_hist = list()
        counter = 0

        training_acc = list()
        testing_acc = list()

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

                                spiked_sample = spiker(sample, num_steps=num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)

                        spike_record, memory_record, _ = net(spiked_data, targets, False)

                        
                        loss_val = loss(spike_record, targets)
                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()

                        acc = snnfunc.acc.accuracy_rate(spike_record, targets)
                        training_acc.append(acc)
                        



                        loss_hist.append(loss_val.item())


                        with torch.no_grad():

                                net.eval()
                                test_data, test_targets = next(iter(test_loader))
                                test_data = test_data.to(device)
                                test_targets = test_targets.to(device)
                                test_targets = test_targets

                                spiked_test_data = list()
                                for sample in test_data.view(batch_size, -1):

                                        spiked_sample = spiker(sample, num_steps=num_steps)

                                        spiked_test_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                                spiked_test_data = torch.stack(spiked_test_data).to(device)

                                
                                test_spike, test_memory, sample = net(spiked_test_data, test_targets, False)

                                
                                test_loss = loss(test_spike, test_targets)

                                test_acc = snnfunc.acc.accuracy_rate(test_spike, test_targets)
                                testing_acc.append(test_acc)
                                
                                #print(f"Number killed: {num_killed}")
                                #print(f"Accuracy at testing: {int(test_acc * batch_size)}/{batch_size}, {test_acc * 100}%\n")
                                #
                                test_loss_hist.append(test_loss.item())


                                if counter % 50 == 0:
                                        train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter)
                                        

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

                                spiked_sample = spiker(sample, num_steps=num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)


                        test_spikes, _, _ = net(spiked_data, targets, False)

                        _, predicted = test_spikes.sum(dim=0).max(1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

        
        
        performance = 100 * correct / total
        print("Performance:", performance)


        net(spiked_data, targets, True)

        
                                
if __name__ == '__main__':
        main()
        #cProfile.run(main(), filename = 'profiling.txt')
