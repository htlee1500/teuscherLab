# imports
import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import Fractional_LIF_P
import Fractional_LIF
import numpy as np
import util

import timeit
import random
import plotsrc
import matplotlib.pyplot as plt

class SNN(nn.Module):
        def __init__(self, num_input, num_hidden, num_output, num_steps):
                super().__init__()

                self.hidden_synapses = nn.Linear(num_input, num_hidden, bias = False)
                self.hidden_synapses.weight = nn.Parameter(torch.abs(torch.mul(self.hidden_synapses.weight, 6)))
                #self.hidden_synapses.weight = nn.Parameter(torch.mul(self.hidden_synapses.weight, 10))
                self.flif_hidden = Fractional_LIF_P.FLIFP(num_hidden)
                self.output_synapses = nn.Linear(num_hidden, num_output, bias = False)
                self.output_synapses.weight = nn.Parameter(torch.abs(torch.mul(self.output_synapses.weight, 5)))
                #self.output_synapses.weight = nn.Parameter(torch.mul(self.output_synapses.weight, 3))
                self.flif_output = Fractional_LIF.FLIF(num_output)
                self.num_steps = num_steps

                self.frac_order = self.flif_hidden.alpha
                self.num_input = num_input
                self.num_hidden = num_hidden
                self.num_output = num_output


        def forward(self, data):

                start = timeit.default_timer()

                output_mem = self.flif_output.init_mem(data.size(0))
                output_spikes_trace = list()
                output_values_trace = list()
                
                hidden_current = self.hidden_synapses(data)
                hidden_spikes, hidden_mem = self.flif_hidden(hidden_current)

                halfway = timeit.default_timer()
                print("Hidden layer finished in", halfway-start, "seconds")

                output_current_trace = list()
                
                for step in range(self.num_steps):

                        output_current = self.output_synapses(hidden_spikes[:,step,:])
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_spikes_trace.append(output_spikes)
                        output_values_trace.append(output_mem)
                        output_current_trace.append(output_current)


                end = timeit.default_timer()
                print("Output layer finished in", end-halfway, "seconds. Batch processed in", end-start, "seconds")


                sample = random.randint(0, 127)

                figure, ax = plt.subplots(2)

                for i in range(10):
                        neuron = random.randint(0, self.flif_hidden.layer_size-1)
                        ax[0].plot(np.arange(self.num_steps), hidden_mem[sample,:, neuron].tolist(), label = str(neuron))
                        ax[1].plot(np.arange(self.num_steps), hidden_current[sample,:, neuron].tolist(), label = "Current to " + str(neuron))
                        
                ax[0].set_title("Hidden Layer for Sample " + str(sample))
                ax[0].legend()
                ax[1].set_title("Hidden Layer inputs")
                ax[1].legend()
                plt.show()

                figure, ax = plt.subplots(2)

                output_values_trace = torch.stack(output_values_trace)
                output_current_trace = torch.stack(output_current_trace)
                output_spikes_trace = torch.stack(output_spikes_trace)

                for i in range(10):
                        ax[0].plot(np.arange(self.num_steps), output_values_trace[:,sample, i].tolist(), label = str(i))
                        ax[1].plot(np.arange(self.num_steps), output_current_trace[:,sample, i].tolist(), label = "Current to " + str(i))

                ax[0].set_title("Output Layer for Sample " + str(sample))
                ax[0].legend()
                ax[1].set_title("Output Layer inputs")
                ax[1].legend()
                plt.show()
                        
                plotsrc.plot_snn_spikes(data[sample,:, :], data.size(2), hidden_spikes[:,sample, :], output_spikes_trace[:, sample, :], num_steps = self.num_steps,  title="Sample no. "+ str(sample))
                
                quit()
                return output_spikes_trace, output_values_trace



        
#TODO: Adapt with spiking
class SNN2(nn.Module):


        def __init__(self, num_input, num_hidden1, num_hidden2, num_output, num_steps):
                super().__init__()

                self.hidden1_synapses = nn.Linear(num_input, num_hidden1)
                self.flif_hidden1 = Fractional_LIF.FLIF(num_hidden1)

                self.hidden2_synapses = nn.Linear(num_hidden1, num_hidden2)
                self.flif_hidden2 = Fractional_LIF.FLIF(num_hidden2)

                self.output_synapses = nn.Linear(num_hidden2, num_output)
                self.flif_output = Fractional_LIF.FLIF(num_output)
                self.num_steps = num_steps

                self.frac_order = self.flif_hidden1.alpha
                self.num_input = num_input
                self.num_hidden1 = num_hidden1
                self.num_hidden2 = num_hidden2
                self.num_output = num_output

        def forward(self, data):
                """
                temp = data.tolist()
                for x in range(28):
                        string = ""
                        for y in range(28):
                                string += str(temp[0][x+y])#str(round(temp[0][x+y] * 10)/10)
                                string += " "
                        print(string)
                        print("\n")
                """

                # initialize values
                hidden1_mem = self.flif_hidden1.init_mem()
                hidden2_mem = self.flif_hidden2.init_mem()
                output_mem = self.flif_output.init_mem()


                # Track outputs
                output_spikes_trace = list()
                output_values_trace = list()


                for step in range(self.num_steps):

                        hidden1_current = self.hidden1_synapses(data)
                        hidden1_spikes, hidden1_mem = self.flif_hidden1(hidden1_current, hidden1_mem)

                        hidden2_current = self.hidden2_synapses(hidden1_spikes)
                        hidden2_spikes, hidden2_mem = self.flif_hidden2(hidden2_current, hidden2_mem)
                        
                        output_current = self.output_synapses(hidden2_spikes)
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_spikes_trace.append(output_spikes)
                        output_values_trace.append(output_mem)


                return torch.stack(output_spikes_trace, dim=0), torch.stack(output_values_trace, dim=0)


        def get_dims(self):
                return "(" + str(self.num_input) + "x" + str(self.num_hidden1) + "x" + str(self.num_hidden2) + "x" +  str(self.num_output) + ")"
