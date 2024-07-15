# imports
import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import plotsrc

import random


class NN(nn.Module):
        def __init__(self, beta, num_input, num_hidden, num_output, num_steps):
                super().__init__()

                self.hidden_synapses = nn.Linear(num_input, num_hidden)
                self.flif_hidden = snn.Leaky(beta=beta)
                self.output_synapses = nn.Linear(num_hidden, num_output)
                self.flif_output = snn.Leaky(beta=beta)
                self.num_steps = num_steps

                

        def forward(self, data, testing=False):

                # initialize values
                hidden_mem = self.flif_hidden.init_leaky()
                output_mem = self.flif_output.init_leaky()


                # Track outputs
                output_spikes_trace = list()
                output_values_trace = list()

                hidden_spikes_trace = list()
                hidden_mem_trace = list()

                hidden_current_trace = list()
                output_current_trace = list()

                spiked_data = list()
                for sample in data:

                        spiked_sample = spikegen.rate(sample, num_steps = self.num_steps)
                        
                        

                        spiked_data.append(spiked_sample) # contains batch_size num_steps x input_size tensors (for MNIST, 128 25x784 tensors)

                #spiked_data = torch.transpose(torch.stack(spiked_data), 1, 0)
                spiked_data = torch.stack(spiked_data)
                for step in range(self.num_steps):

                        #torch.set_printoptions(threshold=10_000)
                        #rint(spiked_data[0,step,:])
                        hidden_current = self.hidden_synapses(spiked_data[:,step,:])
                        hidden_spikes, hidden_mem = self.flif_hidden(hidden_current, hidden_mem)
                        
                        hidden_mem_trace.append(hidden_mem)
                        hidden_current_trace.append(hidden_current)
                        hidden_spikes_trace.append(hidden_spikes)

                        output_current = self.output_synapses(hidden_spikes)
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_current_trace.append(output_current)
                        output_spikes_trace.append(output_spikes)
                        output_values_trace.append(output_mem)

                hidden_spikes_trace = torch.stack(hidden_spikes_trace)
                output_spikes_trace = torch.stack(output_spikes_trace)

                #print
                
                sample = random.randint(0, data.size()[0]-1) # batch_size
                if testing:

                        hidden_mem_trace = torch.stack(hidden_mem_trace)
                        hidden_current_trace = torch.stack(hidden_current_trace)
                        output_current_trace = torch.stack(output_current_trace)

                        figure, ax = plt.subplots(2)

                        for i in range(10):
                                neuron = random.randint(0, self.flif_hidden.layer_size-1)
                                ax[0].plot(np.arange(self.num_steps), hidden_mem_trace[:,sample, neuron].tolist(), label = str(neuron))
                                ax[1].plot(np.arange(self.num_steps), hidden_current_trace[:,sample, neuron].tolist(), label = "Current to " + str(neuron))
                        ax[0].set_title("Hidden Layer for Sample " + str(sample))
                        ax[0].legend()
                        ax[1].set_title("Hidden Layer inputs")
                        ax[1].legend()
                        plt.show()

                        figure, ax = plt.subplots(2)

                        for i in range(10):
                                ax[0].plot(np.arange(self.num_steps), output_values_trace[:,sample, i].tolist(), label = str(i))
                                ax[1].plot(np.arange(self.num_steps), output_current_trace[:,sample, i].tolist(), label = "Current to " + str(i))

                        ax[0].set_title("Output Layer for Sample " + str(sample))
                        ax[0].legend()
                        ax[1].set_title("Output Layer inputs")
                        ax[1].legend()
                        plt.show()
                        

                        plotsrc.plot_snn_spikes(spiked_data[sample], hidden_spikes_trace[:,sample,:], output_spikes_trace[:,sample,:], self.num_steps, "Spikes at step " + str(sample))


                return output_spikes_trace, torch.stack(output_values_trace, dim=0)

