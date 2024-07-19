# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import math
import datetime

# Standard LIF model; takes in a voltage value and calculates
# the voltage at the next time step.
# Return types: spike - {0, 1}, V - float
def leaky_integrate_neuron(V, I, time_step = 0.1, thresh=-50, V_reset=-70, gl=0.025, Cm=0.5):

	spike = (V > thresh)

	tau = Cm / gl
	V = V + (time_step/tau)*(-1 * V + I/gl)

	V -= (thresh - V_reset)*spike

	return spike, V

# Fractional LIF model; uses previous voltage values to approximate next voltage
# Return types: spike - {0,1}, V_new - float
def frac_num_lif(V_trace, V_weight, I=3, thresh=-50, V_reset=-70, Vl=-70, dt=0.1, beta=0.2, gl=0.025, Cm=0.5):

	N = len(V_trace)
	V = V_trace[N - 1]
	tau = Cm / gl

	spike = (V > thresh)

	# V_new is the voltage at t_N+1
	# Computing Markov term
	V_new = dt**(beta) * math.gamma(2-beta) * (-gl*(V-Vl)+I) / Cm + V

	# Computing voltage trace

	voltage_trace = 0
	# Trace calculated via inner product of voltage delta and weight vectors
	delta_V = np.subtract(V_trace[1:],V_trace[0:(len(V_trace)-1)])
	memory_V = np.inner(V_weight[-len(V_trace)+1:],delta_V[0:(len(delta_V))])

	V_new -= memory_V

	# Reset voltage if spiking (not sure if this is computationally efficient)
	V_new -= (thresh - V_reset)*spike

	return spike, V_new

# Iterates a layer of FLIF neurons based on given input currents and history
# Takes in layer trace tensor (num_steps x layer size), weight vector, currents vector from previous layer (layer size)
# Can pass in more parameters later for changing model
# Outputs: spike vector (entries in {0, 1}), new voltage vector (layer size)
def frac_lif_layer(step, trace, weight_vector, currents):

	new_voltages = list()
	spikes = list()

	layer_size = trace.size()[1]
	# iterate over trace columns
	# for each neuron in the layer
	# run frac_num_lif using trace, weight vector, input current
	currents_as_list  = currents.tolist()[0]
	# For first step, x
	if step < 2:

		for i in range(layer_size):

			prev_voltage = trace[step, i].item()

			I = 4 + currents_as_list[i]


			spike, new_voltage = leaky_integrate_neuron(prev_voltage, I)

			spikes.append(0 + 1*spike)
			new_voltages.append(new_voltage)

	else:

		for i in range(layer_size):

			voltage_trace = trace[0:step, i].tolist()
			#print(len(voltage_trace))
			I = 4 + currents_as_list[i]

			spike, new_voltage = frac_num_lif(voltage_trace, weight_vector, I)

			spikes.append(0 + 1 * spike)
			new_voltages.append(new_voltage)


	return spikes, new_voltages

# MAIN FUNCTION
def main():

	num_steps = 2000
	beta = 0.2
	# initial voltage, -70 millivolts
	V = -70
	V_trace = list() # list captures all computed voltage values
	nv = np.arange(num_steps-1)
	weight_vector = (num_steps+1-nv)**(1-beta)-(num_steps-nv)**(1-beta)

	# Layers
	num_input = 10
	num_hidden = 10
	num_output = 10

	# Connect neuron layers
	conn_input = nn.Linear(num_input, num_hidden)
	conn_output = nn.Linear(num_hidden, num_output)

	# Input for training
        # Random values for dummy - get dataset later?
	input_spikes = spikegen.rate_conv(torch.rand((num_steps, num_input))).unsqueeze(1)


	# Every neuron needs its own history vector - build a matrix for each hidden layer + output layer
	# Matrix dimensions are [num_steps, layer size]
	# Init values for neurons: use -70mV globally for now
	hidden_layer_trace = torch.ones((2,), dtype = torch.float64)
	hidden_layer_trace = hidden_layer_trace.new_full((num_steps, num_hidden), V)

	output_layer_trace = torch.ones((2,), dtype = torch.float64)
	output_layer_trace = output_layer_trace.new_full((num_steps, num_output), V)


	# Spike trackers
	hidden_spikes = list()
	output_spikes = list()


	# for each step:
		# for each layer:
			# compute input currents
			# iterate all neurons in layer given input currents

	for step in range(num_steps):


		hidden_current = conn_input(input_spikes[step]) # This is a tensor (1 x layer size)
		step_hidden_spikes, step_hidden_trace = frac_lif_layer(step, hidden_layer_trace, weight_vector, hidden_current)

		# Add step_hidden_trace to full trace
		hidden_layer_trace[step] = torch.tensor(step_hidden_trace)

		output_current = conn_output(torch.tensor(step_hidden_spikes, dtype = torch.float32).unsqueeze(0))
		step_output_spikes, step_output_trace = frac_lif_layer(step, output_layer_trace, weight_vector, output_current)

		output_layer_trace[step] = torch.tensor(step_output_trace)


		hidden_spikes.append(torch.tensor(step_hidden_spikes, dtype = torch.float32))
		output_spikes.append(torch.tensor(step_output_spikes, dtype = torch.float32))


	hidden_spikes = torch.stack(hidden_spikes)
	output_spikes = torch.stack(output_spikes)


	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for step in range(int(num_steps)):
		f.write(str(hidden_layer_trace[step].tolist()))
		f.write("\n")
	f.close()


	plotsrc.plot_snn_spikes(input_spikes, hidden_spikes, output_spikes, num_steps, "Toy FLIF SNN")


if __name__ == '__main__':
	main()
