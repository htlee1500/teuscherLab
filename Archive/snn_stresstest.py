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

import Fractional_LIF
import flif_snn
import random


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
        
        


        num_input = scale*scale
        num_hidden1 = 1000
        num_hidden2 = 750
        num_output = 10

        num_steps = 200


        net = flif_snn.SNN(num_input, num_hidden1, num_output, num_steps, device).to(device)

        loader = np.load("MNIST_Training/parameters.npz", allow_pickle = True)
        hid_con = loader['hid_con']
        out_con = loader['out_con']

        hid_con = torch.tensor(hid_con).to(device)
        out_con = torch.tensor(out_con).to(device)

        success = net.load(hid_con, out_con)
        
        if success == -1:
                quit()

        num_epochs = 1
        unperturbed_accuracy = list()
        noisy_accuracy = list()
        damaged_accuracy = dict()


        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

        # Measure accuracy without changes
        for epoch in range(num_epochs):
                
                iter_counter = 0
                test_batch = iter(test_loader)

                plot = False

                
                for data, targets in test_batch:


                        
                        data = data.to(device)
                        targets = targets.to(device)
                        targets = targets
                        
                        net.train()

                        spiked_data = list()

                        #print(data.view(batch_size, -1).size())
                        
                        for sample in data.view(batch_size, -1):

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)

                        spike_record, memory_record, _ = net(spiked_data, targets, False)


                        acc = snnfunc.acc.accuracy_rate(spike_record, targets)
                        unperturbed_accuracy.append(acc)


        unperturbed_accuracy = np.sum(unperturbed_accuracy) / len(unperturbed_accuracy)
        print(f"Base accuracy: {unperturbed_accuracy}")
        # Measure accuracy with noise

        
        
        for epoch in range(num_epochs):
                
                iter_counter = 0
                test_batch = iter(test_loader)

                plot = False

                
                for data, targets in test_batch:


                        
                        data = data.to(device)
                        targets = targets.to(device)
                        targets = targets
                        
                        net.train()

                        spiked_data = list()
                        for sample in data.view(batch_size, -1):

                                # Introduce noise

                                spiked_sample = spikegen.rate(sample, num_steps = num_steps)

                                spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                        spiked_data = torch.stack(spiked_data)

                        spike_record, memory_record, _ = net(spiked_data, targets, False)


                        acc = snnfunc.acc.accuracy_rate(spike_record, targets)
                        noisy_accuracy.append(acc)

        noisy_accuracy = np.sum(noisy_accuracy) / len(noisy_accuracy)
        
        print(f"Accuracy after introducing noise: {noisy_accuracy}")
        
        # Measure accuracy with disconnected neuron(s)
        """
        hid_dead = torch.clone(hid_con)
        out_dead = torch.clone(out_con)
        """
        killed = list()
        for i in range(20):
                # Choose a hidden neuron to 'kill'
                
                dead = random.randint(10, 90)
                while dead in killed:

                        dead = random.randint(10, 90)
                        
                killed.append(dead)
                #hid_dead[dead, :] = torch.zeros_like(hid_dead[dead, :]) # hidden x input
                #out_dead[:, dead] = torch.zeros_like(out_dead[:, dead]) # output x hidden
                
                #net.load(hid_dead, out_dead)
                
                
                missing_neuron = list()
        
                for epoch in range(num_epochs):
                        
                        iter_counter = 0
                        test_batch = iter(test_loader)

                        plot = False

                
                        for data, targets in test_batch:


                        
                                data = data.to(device)
                                targets = targets.to(device)
                                targets = targets
                        
                                net.train()

                                spiked_data = list()
                                for sample in data.view(batch_size, -1):

                                        spiked_sample = spikegen.rate(sample, num_steps = num_steps)
                                        
                                        #spiked_sample[:, dead] = torch.zeros_like(spiked_sample[:, dead])
                                        for neuron in killed:

                                                spiked_sample[:, neuron] = torch.zeros_like(spiked_sample[:, neuron])

                                        spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                                spiked_data = torch.stack(spiked_data)

                                spike_record, memory_record, _ = net(spiked_data, targets, False)


                                acc = snnfunc.acc.accuracy_rate(spike_record, targets)
                                missing_neuron.append(acc)

                missing_neuron = np.sum(missing_neuron) / len(missing_neuron)
                print(f"Accuracy after killing input neuron(s) {killed}: {missing_neuron}")
                damaged_accuracy.update({dead : missing_neuron})

if __name__ == '__main__':
        main()
