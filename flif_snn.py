# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import Fractional_LIF
import Fractional_LIF_test
import math
import random

import timeit


class SNN(nn.Module):
        def __init__(self, num_input, num_hidden, num_output, num_steps, device):
                super().__init__()

                self.hidden_synapses = nn.Linear(num_input, num_hidden)
                self.flif_hidden = Fractional_LIF.FLIF(num_hidden, device, num_steps)
                """
                self.inhib_synapses = nn.Linear(num_hidden, num_hidden)
                weight = self.inhib_synapses.weight
                weight = torch.abs(weight)
                weight = torch.mul(weight, -1)
                weight.fill_diagonal_(0)
                self.inhib_synapses.weight = nn.Parameter(weight)
                """
                self.output_synapses = nn.Linear(num_hidden, num_output)
                self.flif_output = Fractional_LIF.FLIF(num_output, device, num_steps)
                
                self.num_steps = num_steps
                self.device = device


        def forward(self, data, targets, plotting):

                # initialize values
                plotting = plotting

                

                hidden_mem = self.flif_hidden.init_mem(data.size(0))
                output_mem = self.flif_output.init_mem(data.size(0))


                # Track outputs
                output_spikes_trace = list()
                output_mem_trace = list()

                hidden_spikes_trace = list()
                hidden_mem_trace = list()
                input_current_trace = list()
                hidden_current_trace = list()

                #inhib_current = torch.zeros(0)
                

                for step in range(self.num_steps):

                        hidden_current = self.hidden_synapses(data[:,step,:])
                        """
                        if step > 0:
                                hidden_current = hidden_current + inhib_current
                        """
                        hidden_spikes, hidden_mem = self.flif_hidden(hidden_current, hidden_mem)

                        #inhib_current = self.inhib_synapses(hidden_spikes)

                        hidden_mem_trace.append(hidden_mem)
                        input_current_trace.append(hidden_current)
                        hidden_spikes_trace.append(hidden_spikes)

                        output_current = self.output_synapses(hidden_spikes)
                        #output_current = torch.add(output_current, 1)
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_spikes_trace.append(output_spikes)
                        hidden_current_trace.append(output_current)
                        output_mem_trace.append(output_mem)

                        
                sample = random.randint(0, data.size(0)-1)

                self.hid_spk = torch.stack(hidden_spikes_trace)
                
                if plotting:
                        
                        """
                        hidden_spikes_trace = torch.stack(hidden_spikes_trace)
                        output_spikes_thing = torch.stack(output_spikes_trace)

                        hidden_mem_trace = torch.stack(hidden_mem_trace)
                        output_mem_trace = torch.stack(output_values_trace)

                        input_current_trace = torch.stack(input_current_trace)
                        hidden_current_trace = torch.stack(hidden_current_trace)
                

                        figure, ax = plt.subplots(2)

                        for i in range(10):
                                neuron = random.randint(0, self.flif_hidden.layer_size-1)
                                ax[0].plot(np.arange(self.num_steps), hidden_mem_trace[:,sample, neuron].tolist(), label = str(neuron))
                                ax[1].plot(np.arange(self.num_steps), input_current_trace[:,sample, neuron].tolist(), label = "Current to " + str(neuron))
                        ax[0].set_title("Hidden Layer for Sample " + str(sample))
                        ax[0].legend()
                        ax[1].set_title("Hidden Layer inputs")
                        ax[1].legend()
                        plt.show()

                        figure, ax = plt.subplots(2)

                        for i in range(10):
                                ax[0].plot(np.arange(self.num_steps), output_mem_trace[:,sample, i].tolist(), label = str(i))
                                ax[1].plot(np.arange(self.num_steps), hidden_current_trace[:,sample, i].tolist(), label = "Current to " + str(i))

                        ax[0].set_title("Output Layer for Sample " + str(sample))
                        ax[0].legend()
                        ax[1].set_title("Output Layer inputs")
                        ax[1].legend()
                        plt.show()
                        
                        plotsrc.plot_snn_spikes(data[sample,:, :], data.size(2), hidden_spikes_trace[:,sample, :], output_spikes_thing[:, sample, :], num_steps = self.num_steps,  title="Sample no. "+ str(sample))
                        """
                        hid_spk = torch.stack(hidden_spikes_trace).detach().cpu().numpy()
                        hid_mem = torch.stack(hidden_mem_trace).detach().cpu().numpy()
                        out_spk = torch.stack(output_spikes_trace).detach().cpu().numpy()
                        out_mem = torch.stack(output_mem_trace).detach().cpu().numpy()
                        tar = targets.detach().cpu().numpy()
                        in_spk = data.detach().cpu().numpy()

                        file_location = "../MNIST_Training/post_train_ce" + ".npz"
                        np.savez(file_location, hid_spk=hid_spk, hid_mem=hid_mem, out_spk=out_spk, out_mem=out_mem, tar=tar, in_spk=in_spk)


                        hid_con = self.hidden_synapses.weight.detach().cpu().numpy()
                        out_con = self.output_synapses.weight.detach().cpu().numpy()

                        np.savez("../MNIST_Training/parameters_rat.npz", hid_con=hid_con, out_con=out_con)
                        

                        
                return torch.stack(output_spikes_trace, dim=0), torch.stack(output_mem_trace, dim=0), sample

        def load(self, hidden, output):

                self.hidden_synapses.weight = nn.Parameter(hidden)
                self.output_synapses.weight = nn.Parameter(output)
        
        # Do not use; for sanity purposes only
        def test(self, data):

                sample = 37
                neuron = 345
                hidden_mem = self.flif_hidden.init_mem(data.size(0))

                fractest = Fractional_LIF_test.Frac_Test(self.num_steps)

                actual_trace = list()
                
                for step in range(self.num_steps):

                        hidden_current = self.hidden_synapses(data[:,step,:])
                        _, hidden_mem = self.flif_hidden(hidden_current, hidden_mem)
                        actual_trace.append(hidden_mem[sample, neuron].item())

                        
                        test_mem = fractest.run(hidden_current[sample, neuron].item())

                #figure, ax = plot.subplots(2)
                
                plt.plot(np.arange(self.num_steps), actual_trace, label = "Network Neuron")
                plt.plot(np.arange(self.num_steps), fractest.V_trace, label = "Free Neuron")
                plt.legend()
                plt.show()
