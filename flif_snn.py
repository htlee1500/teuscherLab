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
import math
import datetime


class SNN(nn.Module):
        def __init__(self, num_input, num_hidden, num_output, num_steps):
                super().__init__()

                self.hidden_synapses = nn.Linear(num_input, num_hidden)
                self.flif_hidden = Fractional_LIF.FLIF(num_hidden)
                self.output_synapses = nn.Linear(num_hidden, num_output)
                self.flif_output = Fractional_LIF.FLIF(num_output)
                self.num_steps = num_steps


        def forward(self, data):

                # initialize values
                hidden_mem = self.flif_hidden.init_mem()
                output_mem = self.flif_output.init_mem()


                # Track outputs
                output_spikes_trace = list()
                output_values_trace = list()


                for step in range(self.num_steps):

                        hidden_current = self.hidden_synapses(data)
                        hidden_spikes, hidden_mem = self.flif_hidden(hidden_current, hidden_mem)

                        output_current = self.output_synapses(hidden_spikes)
                        output_spikes, output_mem = self.flif_output(output_current, output_mem)

                        output_spikes_trace.append(output_spikes)
                        output_values_trace.append(output_mem)


                return torch.stack(output_spikes_trace, dim=0), torch.stack(output_values_trace, dim=0)

