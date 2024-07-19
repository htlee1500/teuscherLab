# temporary storage for state of flif_snn.py

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
                self.output_synapses = nn.Linear(num_hidden, num_output)

                self.feedback = nn.Linear(num_output, num_hidden)

                
                self.flif_hidden = Fractional_LIF.FLIF(num_hidden, device, num_steps)
                self.flif_output = Fractional_LIF.FLIF(num_output, device, num_steps)
                self.num_steps = num_steps
                self.device = device


        def forward(self, data, plotting):

                # initialize values

                spiked_data = list()

                

                hidden_mem = self.flif_hidden.init_mem(data.size(0))
                output_mem = self.flif_output.init_mem(data.size(0))


                # Track outputs
                output_spikes_trace = list()
                output_mem_trace = list()

                hidden_spikes_trace = list()
                hidden_mem_trace = list()
                input_current_trace = list()
                hidden_current_trace = list()
                


                start = timeit.default_timer()
                for step in range(self.num_steps):

                        hidden_current = self.hidden_synapses(data[:,step,:])
                        hidden_spikes, hidden_mem = self.flif_hidden(hidden_current, hidden_mem)

                        hidden_mem_trace.append(hidden_mem)
                        input_current_trace.append(hidden_current)
                        hidden_spikes_trace.append(hidden_spikes)

                        output_current = self.output_synapses(hidden_spikes)
                        #output_current = torch.add(output_current, 1)
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_spikes_trace.append(output_spikes)
                        hidden_current_trace.append(output_current)
                        output_mem_trace.append(output_mem)


                end = timeit.default_timer()
                #print("Batch processed in ", end-start, "seconds")

                sample = random.randint(0, data.size(0)-1)
                
                if plotting:
                        
                        hid_spk = torch.stack(hidden_spikes_trace).detach().cpu().numpy()
                        hid_mem = torch.stack(hidden_mem_trace).detach().cpu().numpy()
                        out_spk = torch.stack(output_spikes_trace).detach().cpu().numpy()
                        out_mem = torch.stack(output_mem_trace).detach().cpu().numpy()

                        file_location = "MNIST_Training/post_train" + ".npz"
                        np.savez(file_location, hid_spk=hid_spk, hid_mem=hid_mem, out_spk=out_spk, out_mem=out_mem)
                        

                        
                return torch.stack(output_spikes_trace, dim=0), torch.stack(output_mem_trace, dim=0), sample

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
