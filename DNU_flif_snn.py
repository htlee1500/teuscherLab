# imports
import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import Fractional_LIF
import numpy as np
import util

import timeit

class SNN(nn.Module):
        def __init__(self, num_input, num_hidden, num_output, num_steps):
                super().__init__()

                self.hidden_synapses = nn.Linear(num_input, num_hidden)
                self.flif_hidden = Fractional_LIF.FLIF(num_hidden, "hidden1")
                self.output_synapses = nn.Linear(num_hidden, num_output)
                self.flif_output = Fractional_LIF.FLIF(num_output, "output")
                self.num_steps = num_steps

                self.frac_order = self.flif_hidden.alpha
                self.num_input = num_input
                self.num_hidden = num_hidden
                self.num_output = num_output


        def forward(self, data):


                # Track outputs
                start = timeit.default_timer()
                
                output_spikes_trace = list()
                output_values_trace = list()

                self.flif_hidden.reset()
                self.flif_output.reset()



                hidden_mem = self.flif_hidden.init_mem()
                output_mem = self.flif_output.init_mem()
                

                tau = self.flif_hidden.tau

                # data is a batch_size x layer_size tensor.
                # Need to turn each sample into spikes. Calling spikegen.latency on data blends samples together, making them unintelligble and worthless
                spiked_data = list()

                created = False

                
                for sample in data:

                        #spiked_sample = spikegen.latency(sample, num_steps = self.num_steps, tau=tau, normalize = True) # Try changing linear=True later; threshold?
                        spiked_sample = spikegen.rate(sample, num_steps = self.num_steps, gain = 8, offset = 3)

                        if created:
                                spiked_data = torch.cat((spiked_data, spiked_sample), dim=0)
                        else:
                                spiked_data = spiked_sample
                                created = True

                        
                batch_size = int(spiked_data.size()[0] / self.num_steps)
                # spiked_data is a num_steps*batch_size x input_size
                # hidden_current SHOULD be a num_steps*batch_size x layer_size tensor
                # Transpose to make it layer_size x num_steps*batch_size
                hidden_current = torch.transpose(self.hidden_synapses(spiked_data), 1, 0)
                # hidden_spikes, hidden_mem are layer_size x num_steps*batch_size
                hidden_spikes, hidden_mem_history = self.flif_hidden(hidden_current)
                self.flif_hidden.iterate_grad(hidden_mem, torch.transpose(hidden_mem_history, 1, 0), self.num_steps, batch_size)

                
                # output_current is output_layer_size x num_steps*batch_size
                output_current = self.output_synapses(torch.transpose(hidden_spikes, 1, 0))

                # gets a num_steps*batch_size x output_layer_size tensor input, should spit out tensor with same dims
                output_spikes, output_mem_history = self.flif_output(torch.transpose(output_current, 1, 0))
                self.flif_output.iterate_grad(output_mem, torch.transpose(output_mem_history, 1, 0), self.num_steps, batch_size)
                # output_spikes and output_mem are output_layer_size x num_steps*batch_size
                # Need to make them into 3d tensors of dim num_steps x batch_size x layer_size

                final_spikes = list()
                final_mem = list()

                
                startpoint = 0
                length = batch_size
                
                for i in range(self.num_steps):
                        
                        spike_slice = torch.narrow(output_spikes, 1, startpoint, length)
                        final_spikes.append(spike_slice)

                        mem_slice = torch.narrow(output_mem_history, 1, startpoint, length)
                        final_mem.append(mem_slice)

                        startpoint += length

                final_spikes = torch.stack(final_spikes, dim=0)
                final_mem = torch.stack(final_mem, dim = 0)
                

                end = timeit.default_timer()
                print("Batch processed in", (end-start), "seconds")
                
                # final_spikes is a num_steps x layer_size x batch_size tensor
                # needs to be step - batch - layer
                return torch.transpose(final_spikes, 2, 1), torch.transpose(final_mem, 2, 1)
                

        def get_dims(self):
                return "(" + str(self.num_input) + "x" + str(self.num_hidden) + "x" + str(self.num_output) + ")"


        
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
