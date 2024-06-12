# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import compute
import math
import datetime

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import Fractional_LIF
import flif_snn



def print_batch_accuracy(net, batch_size, data, targets, train=False):
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
                print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        else:
                print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter, data, targets, test_data, test_targets, net, batch_size):

        print(f"Epoch {epoch}, Iteration {iter_counter}")

        print(f"Train Set Loss: {loss_hist[counter]:.2f}")

        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

        print_batch_accuracy(net, batch_size, data, targets, train=True)

        print_batch_accuracy(net, batch_size, test_data, test_targets, train=False)

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
        num_hidden = 1000
        num_output = 10

        num_steps = 25
        
        net = flif_snn.SNN(num_input, num_hidden, num_output, num_steps).to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas = (0.9, 0.999))


        num_epochs = 5
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
                        
                        spike_record, memory_record = net(data.view(batch_size, -1))

                        loss_val = torch.zeros((1), dtype=dtype, device=device)
                        for step in range(num_steps):
                                loss_val += loss(memory_record[step], targets)


                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()

                        loss_hist.append(loss_val.item())


                        with torch.no_grad():
                                net.eval()
                                test_data, test_targets = next(iter(test_loader))
                                test_data = test_data.to(device)
                                test_targets = test_targets.to(device)

                                
                                #print(test_data.view(batch_size, -1).size())
                                test_spike, test_memory = net(test_data.view(batch_size, -1))

                                test_loss = torch.zeros((1), dtype = dtype, device=device)
                                for step in range(num_steps):
                                        test_loss += loss(test_memory[step], test_targets)

                                test_loss_hist.append(test_loss.item())


                                if counter % 50 == 0:
                                        train_printer(epoch, iter_counter, loss_hist, test_loss_hist, counter, data, targets, test_data, test_targets, net, batch_size)

                                counter += 1
                                iter_counter += 1

        
        
if __name__ == '__main__':
        main()
