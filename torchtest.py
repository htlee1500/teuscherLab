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

# Standard LIF model; takes in a voltage value and calculates
# the voltage at the next time step.
def leaky_integrate_neuron(V, time_step=1e-4, I=0.3e-9, gl=2.5e-8, Cm=5e-10):
	tau = Cm / gl
	V = V + (time_step/tau)*(-1 * V + I/gl)
	return V

# Fractional LIF model; uses previous voltage values to approximate next voltage
def frac_num_lif(V_trace, V_weight, thresh=-50, V_reset=-70,  Vl=-70, dt=0.1, beta=0.2, gl=0.025, Cm=0.5,I=3):

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

	return V_new

# MAIN FUNCTION
def main():

	num_steps = 2000
	beta = 0.2
	# initial voltage, -70 millivolts
	V = -70
	V_trace = list() # list captures all computed voltage values
	nv = np.arange(num_steps-1)
	V_weight = (num_steps+1-nv)**(1-beta)-(num_steps-nv)**(1-beta)

	for step in range(int(num_steps)):

		# First 2 time steps t0, t1 are computed using regular LIF
		if step < 2:

			V_trace.append(V)
			V = leaky_integrate_neuron(V)

		# All others use the fractional LIF
		else:
			V_trace.append(V)
			V = frac_num_lif(V_trace, V_weight)

	for step in range(int(num_steps)):

		V_trace[step] = V_trace[step]*1e-3


	# This block writes the voltage values to a text file
	# for troubleshooting and debugging

	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for step in range(int(num_steps)):
		f.write(str(V_trace[step]) + "\n")

	f.close()

	# Plotting the list of voltage values
	plotsrc.plot_mem(V_trace, 0, num_steps, -0.07, -0.05, "Fractional Leaky Neuron Model", True, str(now))


#def main():

if __name__ == '__main__':
	main()
