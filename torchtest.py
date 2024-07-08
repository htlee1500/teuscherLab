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
import lif_snn

import cProfile


def train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter):

        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print("\n")


# MAIN FUNCTION
def main():
        
        batch_size = 128
        data_path='/tmp/data/mnist'

        dtype = torch.float
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        print(mnist_train)
        
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


        num_input = 28*28
        num_hidden1 = 2000
        num_hidden2 = 750
        num_output = 10

        num_steps = 250

        gain = 1
        
        #net = flif_snn.SNN2(num_input, num_hidden1, num_hidden2, num_output, num_steps).to(device)
        net = flif_snn.SNN(num_input, num_hidden1, num_output, num_steps, gain).to(device)
        #net = lif_snn.NN(.95, num_input, num_hidden1, num_output, num_steps).to(device)
        #loss = nn.CrossEntropyLoss()
        loss = snnfunc.ce_count_loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas = (0.9, 0.999))


        num_epochs = 1
        loss_hist = list()
        test_loss_hist = list()
        counter = 0

        
        for epoch in range(num_epochs):
                iter_counter = 0
                train_batch = iter(train_loader)

                
                for data, targets in train_batch:


                        
                        data = data.to(device)
                        targets = targets.to(device)
                        
                        net.train()

                        spiked_data = list()
                        for sample in data.view(batch_size, -1):

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps, gain = gain)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)
                        
                        spike_record, memory_record = net(spiked_data)

                        _, idx = test_spike.detach().sum(dim=0).max(1)
                        total = targets.size(0)
                        correct = (idx == targets).sum().item()
                        print(correct / total)



                        start = timeit.default_timer()
                        loss_val = loss(spike_record, targets)
                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()
                        end = timeit.default_timer()
                        print("Loss and backprop in", end-start, "seconds")



                        loss_hist.append(loss_val.item())


                        with torch.no_grad():

                                net.eval()
                                test_data, test_targets = next(iter(test_loader))
                                test_data = test_data.to(device)
                                test_targets = test_targets.to(device)

                                spiked_test_data = list()
                                for sample in test_data.view(batch_size, -1):

                                        spiked_sample = spikegen.rate(sample, num_steps = num_steps, gain = gain)

                                        spiked_test_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                                spiked_test_data = torch.stack(spiked_test_data)

                                
                                test_spike, test_memory = net(spiked_test_data)
                                #print("Testing batch processed")


                                #test_loss = torch.zeros((1), dtype = dtype, device=device)

                                _, idx = test_spike.detach().sum(dim=0).max(1)
                                total = targets.size(0)
                                correct = (idx == targets).sum().item()

                                test_loss = loss(test_spike, test_targets)
                                
                                test_loss_hist.append(test_loss.item())


                                if counter % 50 == 0:
                                        train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter)
                                        
                                print("Test set accuracy:", acc)
                                counter += 1
                                iter_counter += 1

        iter_len = np.arange(counter)
        plt.plot(iter_len, loss_hist, label = "Training Loss")
        plt.plot(iter_len, test_loss_hist, label = "Testing Loss")
        plt.legend()
        
        # get overall results for file
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle = True, drop_last = False)

        total = 0
        correct = 0
        with torch.no_grad():
                net.eval()
                for data, targets in test_loader:
                        data = data.to(device)
                        targets = targets.to(device)

                        spiked_data = list()
                        for sample in data.view(data.size(0), -1):

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps, gain = gain)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)


                        test_spikes, _ = net(spiked_data)

                        _, predicted = test_spikes.sum(dim=0).max(1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

        performance = 100 * correct / total
        print("Performance:", performance)

        
        plt.show()
                                
if __name__ == '__main__':
        main()
        #cProfile.run(main(), filename = 'profiling.txt')
